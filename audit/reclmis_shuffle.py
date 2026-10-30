"""Counterfactual-text test for RecLMIS (and other LViT-family repos with the same data layout).

Run from inside the RecLMIS repo:
  cd $EXT/RecLMIS
  python $AUDIT/reclmis_shuffle.py --cfg Config_covid19 \
      --ckpt ckpt/qata/best_model-RecLMIS.pth.tar --out $OUT/reclmis_qata

Conditions (each is a full pass over the test set with RecLMIS' own loader and model call):
  matched     own report
  constant    every image gets the most frequent report (in-distribution, no case info)
  shuffled_sK another image's report with DIFFERENT wording (K = seeds)
  lr_swap     left<->right swapped; scored only on images where that changes the text
  matched_nomask  own report, but the ground-truth mask argument RecLMIS' test code passes to the
                  model replaced by zeros -> checks the test-time forward does not use the label
"""
import argparse
import collections
import csv
import importlib
import json
import os
import re
import sys
import tempfile

import numpy as np


def lr_swap(t):
    return re.sub(r"\b(left|right|Left|Right)\b",
                  lambda m: {"left": "right", "right": "left", "Left": "Right", "Right": "Left"}[m.group(1)], t)


def build_conditions(df, n_seeds, seed=0):
    """df: DataFrame with Image, Description. Returns {cond: {image: description}}, plus lr_swap eligibility."""
    rng = np.random.default_rng(seed)
    own = dict(zip(df.Image, df.Description))
    descs = list(df.Description)
    common = collections.Counter(descs).most_common(1)[0][0]
    conds = {"matched": dict(own), "constant": {k: common for k in own}}
    for s in range(n_seeds):
        d = {}
        for k, t in own.items():
            while True:
                c = descs[rng.integers(len(descs))]
                if c.strip().lower() != t.strip().lower():
                    d[k] = c
                    break
        conds[f"shuffled_s{s}"] = d
    conds["lr_swap"] = {k: lr_swap(t) for k, t in own.items()}
    # meaning-level asymmetry: the set of "<extent> <side> lung" phrases must change under the swap
    # ("all left lung and all right lung" -> "all right lung and all left lung" is the same finding)
    locs = lambda t: frozenset(re.findall(r"\b(\w+)\s+(left|right)\s+lung", t.lower()))
    lr_ok = {k for k, t in own.items() if locs(t) and locs(lr_swap(t)) != locs(t)}
    return conds, lr_ok, common


def forward_logits(model, img, mask, tok, tmask):
    """RecLMIS' forward only defines img_weight/text_weight when its auxiliary reconstructor succeeds
    (a bare `except` disables it otherwise), and mode="test" then crashes on the return. The
    segmentation path does not depend on that branch, so capture the sigmoid output with a hook."""
    box = {}
    h = model.last_activation.register_forward_hook(lambda m, i, o: box.__setitem__("out", o))
    try:
        model(img, mask, tok, tmask, mode="test")
    except UnboundLocalError:
        pass
    finally:
        h.remove()
    if "out" not in box:
        raise RuntimeError("segmentation output was not produced")
    return box["out"]


def paired_boot(d, n=5000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="Config_covid19")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
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
    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    res = model.load_state_dict(ck["state_dict"], strict=False)
    print(f"checkpoint: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys")
    if res.missing_keys or res.unexpected_keys:
        sys.exit(f"ERROR: checkpoint mismatch - {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected "
                 f"keys, e.g. {(list(res.missing_keys) + list(res.unexpected_keys))[:5]}")
    model.eval()

    xlsx = os.path.join(config.test_dataset, "Test_text.xlsx")
    df = pd.read_excel(xlsx)
    conds, lr_ok, common = build_conditions(df, a.seeds)
    print(f"{len(df)} test reports, {df.Description.nunique()} distinct; constant = {common!r}; "
          f"lr_swap changes {len(lr_ok)} reports")

    tf = ValGenerator(output_size=[config.img_size, config.img_size])
    tmpdir = tempfile.mkdtemp()
    per = collections.defaultdict(dict)
    run_list = list(conds) + ["matched_nomask"]
    for cname in run_list:
        tdict = conds["matched" if cname == "matched_nomask" else cname]
        p = os.path.join(tmpdir, f"{cname}.xlsx")
        # Their read_text pads short reports with ' EOF XXX' via chained assignment, which is a silent
        # no-op under pandas>=3 copy-on-write. Pad here with the identical rule so inputs match training.
        padded = [t + " EOF XXX" * (9 - len(t.split())) if len(t.split()) < 9 else t for t in tdict.values()]
        pd.DataFrame({"Image": list(tdict), "Description": padded}).to_excel(p, index=False)
        text = read_text(p)
        bad = [k for k, t in zip(tdict, padded) if text[k] != t]
        if bad:
            sys.exit(f"ERROR: read_text altered {len(bad)} reports, e.g. {bad[0]!r}")
        ds = ImageToImage2D(config.test_dataset, config.task_name, text, tf, image_size=config.img_size,
                            data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, (b, names) in enumerate(dl):
                if a.limit and i >= a.limit:
                    break
                img, lab = b["image"].cuda(), b["label"].cuda()
                tok, tm = b["text_token"].cuda(), b["text_mask"].cuda()
                m_in = torch.zeros_like(lab) if cname == "matched_nomask" else lab
                out = forward_logits(model, img, m_in, tok, tm)
                pr = (out > 0.5).float()[0].cpu().numpy().reshape(config.img_size, config.img_size)
                gt = lab.cpu().numpy().reshape(config.img_size, config.img_size).astype(np.float32)
                d = 2 * np.sum(gt * pr) / (np.sum(gt) + np.sum(pr) + 1e-5)   # their formula
                per[str(names[0])][cname] = float(d)
        print(f"{cname}: mean Dice {np.mean([v[cname] for v in per.values() if cname in v]):.4f}", flush=True)

    names = sorted(per)
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image"] + run_list)
        for n in names:
            w.writerow([n] + [per[n].get(c) for c in run_list])

    col = lambda c, sel=None: np.array([per[n][c] for n in (sel or names)])
    shuf = np.mean([col(f"shuffled_s{s}") for s in range(a.seeds)], 0)
    # loader names are image files ("x.png"); report keys are mask files ("mask_x.png") in LViT-style data
    key_of = lambda n: n if n in conds["matched"] else ("mask_" + n if "mask_" + n in conds["matched"] else None)
    unmatched = [n for n in names if key_of(n) is None]
    if unmatched:
        print(f"WARNING: {len(unmatched)} image names not found among report keys, e.g. {unmatched[:3]}")
    lr_names = [n for n in names if key_of(n) in lr_ok]
    summ = {
        "n": len(names),
        "dice": {c: float(col(c).mean()) for c in run_list},
        "gain_vs_constant": paired_boot(col("matched") - col("constant")),
        "gain_vs_shuffled": paired_boot(col("matched") - shuf),
        "gain_vs_lr_swap_on_asymmetric": paired_boot(col("matched", lr_names) - col("lr_swap", lr_names)) if lr_names else None,
        "n_lr_asymmetric": len(lr_names),
        "label_leak_check_matched_minus_nomask": paired_boot(col("matched") - col("matched_nomask")),
    }
    g = min(summ["gain_vs_constant"], summ["gain_vs_shuffled"], key=lambda x: x[0])
    # legacy one-sided verdict used during the study; the paper's classification is in analysis/make_tables.py
    summ["checkpoint_keys"] = {"missing": 0, "unexpected": 0}
    summ["verdict"] = ("USES TEXT" if g[0] >= 0.01 and g[1] > 0 else
                       "IGNORES TEXT" if g[0] < 0.005 and g[2] < 0.01 else "AMBIGUOUS")
    if abs(summ["label_leak_check_matched_minus_nomask"][0]) > 0.005:
        summ["verdict"] += " | WARNING: test-time forward depends on the ground-truth mask argument"
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
