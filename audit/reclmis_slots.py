"""Slot-wise counterfactuals for template reports (QaTa-COV19 / MosMedData+ style):

  "<Bilateral|Unilateral> pulmonary infection, <N> infected area(s), <extent> <side> lung[ and <extent> <side> lung]."

Each condition changes ONE slot and is scored only on images where that edit changes the report:
  side     left <-> right                         (meaning-level change only, as in lr_swap)
  count    number word -> a different number word (area/areas kept grammatical)
  extent   all/upper/middle/lower -> a different extent, per location phrase
  lat      Bilateral <-> Unilateral word only
Reported effect = Dice(matched) - Dice(edited), paired over the affected images.

  cd $EXT/RecLMIS
  python $AUDIT/reclmis_slots.py --cfg Config_covid19 --ckpt ckpt/qata/best_model-RecLMIS.pth.tar --out $OUT/slots_qata
"""
import argparse
import importlib
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reclmis_shuffle import forward_logits, lr_swap, paired_boot  # noqa: E402

NUMS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
EXTENTS = ["all", "upper", "middle", "lower"]
LOC = re.compile(r"\b(all|upper|middle|lower)\s+(left|right)\s+lung", re.I)


def locs(t):
    return frozenset((e.lower(), s.lower()) for e, s in LOC.findall(t))


def edit_count(t, rng):
    m = re.search(r"\b(" + "|".join(NUMS) + r")\s+infected\s+area(s?)\b", t, re.I)
    if not m:
        return None
    new = rng.choice([n for n in NUMS if n != m.group(1).lower()])
    noun = "area" if new == "one" else "areas"
    return t[:m.start()] + f"{new} infected {noun}" + t[m.end():]


def edit_extent(t, rng):
    if not LOC.search(t):
        return None
    return LOC.sub(lambda m: f"{rng.choice([e for e in EXTENTS if e != m.group(1).lower()])} {m.group(2)} lung", t)


def edit_lat(t):
    if re.search(r"\bBilateral\b", t):
        return re.sub(r"\bBilateral\b", "Unilateral", t)
    if re.search(r"\bUnilateral\b", t):
        return re.sub(r"\bUnilateral\b", "Bilateral", t)
    return None


def edit_side(t):
    s = lr_swap(t)
    return s if locs(t) and locs(s) != locs(t) else None


def build(df, seed=0):
    rng = np.random.default_rng(seed)
    own = dict(zip(df.Image, df.Description))
    conds = {"side": {}, "count": {}, "extent": {}, "lat": {}}
    for k, t in own.items():
        for name, e in (("side", edit_side(t)), ("count", edit_count(t, rng)),
                        ("extent", edit_extent(t, rng)), ("lat", edit_lat(t))):
            if e is not None and e.strip().lower() != t.strip().lower():
                conds[name][k] = e
    return own, conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="Config_covid19")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, os.getcwd())
    config = importlib.import_module(a.cfg)
    from Load_Dataset import ImageToImage2D, ValGenerator
    from nets.RecLMIS import RecLMIS
    from utils import read_text

    model = RecLMIS(config, config.get_ViT_config(), n_channels=config.n_channels, n_classes=config.n_labels).cuda()
    res = model.load_state_dict(torch.load(a.ckpt, map_location="cuda", weights_only=False)["state_dict"], strict=False)
    print(f"checkpoint: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    if res.missing_keys or res.unexpected_keys:
        sys.exit(f"ERROR: checkpoint mismatch - {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected "
                 f"keys, e.g. {(list(res.missing_keys) + list(res.unexpected_keys))[:5]}")
    model.eval()

    df = pd.read_excel(os.path.join(config.test_dataset, "Test_text.xlsx"))
    own, conds = build(df)
    for c, d in conds.items():
        ex = next(iter(d.items())) if d else None
        print(f"{c}: {len(d)} reports changed; e.g. {own[ex[0]]!r} -> {ex[1]!r}" if ex else f"{c}: 0")
    json.dump({c: d for c, d in conds.items()}, open(os.path.join(a.out, "edits.json"), "w"), indent=1)

    tf = ValGenerator(output_size=[config.img_size, config.img_size])
    tmp = tempfile.mkdtemp()
    key_of = lambda n: n if n in own else ("mask_" + n if "mask_" + n in own else None)

    def run(tdict, tag):
        full = dict(own)
        full.update(tdict)
        padded = [t + " EOF XXX" * (9 - len(t.split())) if len(t.split()) < 9 else t for t in full.values()]
        p = os.path.join(tmp, f"{tag}.xlsx")
        pd.DataFrame({"Image": list(full), "Description": padded}).to_excel(p, index=False)
        text = read_text(p)
        ds = ImageToImage2D(config.test_dataset, config.task_name, text, tf, image_size=config.img_size,
                            data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
        out = {}
        with torch.no_grad():
            for i, (b, names) in enumerate(DataLoader(ds, batch_size=1, shuffle=False)):
                if a.limit and i >= a.limit:
                    break
                k = key_of(str(names[0]))
                if tag != "matched" and k not in tdict:
                    continue                      # only re-run images this edit changes
                o = forward_logits(model, b["image"].cuda(), b["label"].cuda(), b["text_token"].cuda(), b["text_mask"].cuda())
                pr = (o > 0.5).float()[0].cpu().numpy().ravel()
                gt = b["label"].numpy().ravel().astype(np.float32)
                out[k] = float(2 * np.sum(gt * pr) / (np.sum(gt) + np.sum(pr) + 1e-5))
        return out

    base = run({}, "matched")
    summ = {"n": len(base), "matched_dice": float(np.mean(list(base.values()))), "slots": {}}
    per_image = {"matched": base}
    for c, d in conds.items():
        r = run(d, c)
        per_image[c] = r
        ks = [k for k in r if k in base]
        if not ks:
            summ["slots"][c] = None
            continue
        m, lo, hi = paired_boot([base[k] - r[k] for k in ks])
        summ["slots"][c] = {"n": len(ks), "dice_matched": float(np.mean([base[k] for k in ks])),
                            "dice_edited": float(np.mean([r[k] for k in ks])), "effect": m, "ci95": [lo, hi]}
        print(f"{c}: n={len(ks)} effect {100 * m:.2f} Dice pts [{100 * lo:.2f}, {100 * hi:.2f}]", flush=True)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    json.dump(per_image, open(os.path.join(a.out, "per_image.json"), "w"))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
