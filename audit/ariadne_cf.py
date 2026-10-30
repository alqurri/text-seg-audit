"""Counterfactual-text + slot tests for Ariadne's Thread (LanGuideMedSeg) on QaTa-COV19 test.
Run from the LanGuideMedSeg repo root. --ckpt none = untrained weights (pipeline smoke only)."""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils.config as config
from utils.dataset import QaTa
from engine.wrapper import LanGuideMedSegWrapper
from reclmis_shuffle import build_conditions, paired_boot
import reclmis_slots as slots


def dice_pass(model, args, csv_path, dev, state):
    ds = QaTa(csv_path=csv_path, root_path=args.test_root_path, tokenizer=args.bert_type,
              image_size=args.image_size, mode="test")
    names, out = list(ds.image_list), []
    with torch.no_grad():
        for (img, txt), gt in DataLoader(ds, batch_size=16, shuffle=False, num_workers=8):
            p = model([img.to(dev), {k: v.to(dev) for k, v in txt.items()}])
            if "sig" not in state:
                state["sig"] = bool(p.min() < 0 or p.max() > 1)
                print(f"output range [{float(p.min()):.3f}, {float(p.max()):.3f}] -> sigmoid: {state['sig']}")
            if state["sig"]:
                p = p.sigmoid()
            p = (p > 0.5).float().flatten(1)
            g = (gt.to(dev) > 0.5).float().flatten(1)
            s = p.sum(1) + g.sum(1)
            out += torch.where(s > 0, 2 * (p * g).sum(1) / s.clamp(min=1), torch.ones_like(s)).cpu().tolist()
    assert len(out) == len(names), (len(out), len(names))
    return dict(zip(names, out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/training.yaml")
    ap.add_argument("--ckpt", default="./save_model/medseg.ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    args = config.load_cfg_from_cfg_file(a.config)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    model = LanGuideMedSegWrapper(args)
    if a.ckpt != "none":
        sd = torch.load(a.ckpt, map_location="cpu")
        print(model.load_state_dict(sd.get("state_dict", sd), strict=True))
    model.to(dev).eval()

    df = pd.read_csv(args.test_csv_path)
    assert {"Image", "Description"} <= set(df.columns), df.columns
    if a.limit:
        df = df.iloc[:a.limit]
    conds, lr_ok, common = build_conditions(df, a.seeds)
    own, sl = slots.build(df)
    for k, v in sl.items():
        conds["slot_" + k] = {**own, **v}
    affected = {"lr_swap": lr_ok, **{"slot_" + k: set(v) for k, v in sl.items()}}
    print("constant report:", common, "| lr-asymmetric:", len(lr_ok), "| slots:", {k: len(v) for k, v in sl.items()})

    res, state = {}, {}
    for c, m in conds.items():
        p = os.path.join(a.out, f"{c}.csv")
        pd.DataFrame({"Image": list(df.Image), "Description": [m[k] for k in df.Image]}).to_csv(p, index=False)
        res[c] = dice_pass(model, args, p, dev, state)
        print(f"{c:16s} Dice {100 * np.mean(list(res[c].values())):.2f}", flush=True)

    mt = res["matched"]
    sh = [c for c in res if c.startswith("shuffled_")]
    res["shuffled_mean"] = {k: float(np.mean([res[c][k] for c in sh])) for k in mt}
    summ = {"n": len(df), "constant_report": common,
            "dice": {c: 100 * float(np.mean(list(r.values()))) for c, r in res.items()}, "delta": {}}
    for c in res:
        if c == "matched":
            continue
        keys = sorted(affected.get(c, mt.keys()))
        if not keys:
            continue
        m, lo, hi = paired_boot([100 * (mt[k] - res[c][k]) for k in keys])
        summ["delta"][c] = {"n": len(keys), "mean": m, "lo": lo, "hi": hi}
        print(f"delta matched-{c:14s} n={len(keys):5d}  {m:6.2f} [{lo:.2f}, {hi:.2f}]")
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(a.out, "per_image.json"), "w"))


if __name__ == "__main__":
    main()
