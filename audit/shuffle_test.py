"""Report-substitution test for TextBraTS: does the released model use the report at test time?

Evaluates the official TextBraTS checkpoint on its Test split under three text conditions:
  matched  - each patient's own report feature (the paper's setting)
  shuffled - another patient's report (fixed derangement, several seeds)
  mean     - the mean report feature over the test set (in-distribution, carries no case info)

Verdict (one-sided rule used during the study; the paper reports the two-sided +-0.5-point equivalence
analysis computed by analysis/tb_delta_ci.py and analysis/make_table1.py).

Run from inside the TextBraTS repo (it imports their utils/), e.g.
  cd $EXT/TextBraTS
  python $AUDIT/shuffle_test.py --data_dir ./data/TextBraTSData \
      --json_list Test.json --ckpt ./runs/TextBraTS/model.pt --out $OUT/textbrats_kt
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REGIONS = ("TC", "WT", "ET")
# Pre-registered decision on mean-over-regions Dice, paired over test patients, cluster = patient.
NEGLIGIBLE_GAIN = 0.005   # matched - best(no-info condition) < 0.5 Dice point and CI upper < 1.0 point
NEGLIGIBLE_UPPER = 0.010
USES_GAIN = 0.010     # >= 1.0 Dice point with CI lower > 0


def build_args(a):
    """Namespace with exactly the fields TextBraTS' test.py passes to get_loader / the model."""
    ns = argparse.Namespace(
        data_dir=a.data_dir, json_list=a.json_list, test_mode=True, fold=0, workers=a.workers,
        distributed=False, batch_size=1, feature_size=48, in_channels=4, out_channels=3,
        spatial_dims=3, dropout_rate=0.0, use_checkpoint=False, a_min=-175.0, a_max=250.0,
        b_min=0.0, b_max=1.0, space_x=1.5, space_y=1.5, space_z=2.0, roi_x=128, roi_y=128, roi_z=128,
        infer_overlap=0.6, RandFlipd_prob=0.2, RandRotate90d_prob=0.2, RandScaleIntensityd_prob=0.1,
        RandShiftIntensityd_prob=0.1, rank=0, world_size=1)
    return ns


def load_components(a):
    if os.environ.get("TBKT_MOCK"):          # tests only
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
        from mock_textbrats import get_loader, make_model
        return get_loader, make_model
    sys.path.insert(0, os.getcwd())
    from utils.data_utils import get_loader  # TextBraTS repo
    from utils.textswin_unetr import TextSwinUNETR

    def make_model(args):
        return TextSwinUNETR(img_size=128, in_channels=4, out_channels=3, feature_size=48,
                             drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0,
                             use_checkpoint=False, text_dim=768)
    return get_loader, make_model


def dice_per_region(pred, gt):
    """pred, gt: bool arrays (3, ...). Returns per-region Dice; empty-empty counts as 1."""
    out = []
    for c in range(pred.shape[0]):
        p, g = pred[c], gt[c]
        s = p.sum() + g.sum()
        out.append(1.0 if s == 0 else float(2 * (p & g).sum() / s))
    return out


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def paired_boot(diff, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--json_list", default="Test.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle_seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    get_loader, make_model = load_components(a)
    args = build_args(a)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing or unexpected:   # their test.py loads strict=False silently; we refuse any partial load
        print("  first keys:", (list(missing) + list(unexpected))[:10])
        sys.exit("ERROR: checkpoint does not match the model exactly")
    model.to(dev).eval()
    loader = get_loader(args)

    # pass 1: collect text features (and ids) to build no-information conditions
    texts, ids = [], []
    for i, b in enumerate(loader):
        if a.limit and i >= a.limit:
            break
        texts.append(b["text_feature"].float().clone())
        meta = b.get("image_meta_dict", {}) or {}
        fn = meta.get("filename_or_obj", [str(i)])
        ids.append(os.path.basename(str(fn[0] if isinstance(fn, (list, tuple)) else fn)))
    n = len(texts)
    shapes = {tuple(t.shape) for t in texts}
    if len(shapes) != 1:
        sys.exit(f"ERROR: text features have varying shapes {shapes}; mean condition undefined")
    mean_text = torch.stack(texts).mean(0)
    rng = np.random.default_rng(0)
    perms = [derangement(n, rng) for _ in range(a.shuffle_seeds)]
    sim = torch.nn.functional.cosine_similarity(torch.stack(texts).flatten(1),
                                                torch.stack(texts).flatten(1)[torch.as_tensor(perms[0])], dim=1)
    print(f"{n} test cases, text feature shape {shapes.pop()}, "
          f"cosine(own, shuffled) median {sim.median():.3f}")

    conds = ["matched", "mean"] + [f"shuffled_s{k}" for k in range(a.shuffle_seeds)]
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n:
                break
            img = b["image"].to(dev)
            gt = b["label"].numpy()[0].astype(bool)
            if not torch.allclose(b["text_feature"].float(), texts[i]):
                sys.exit("ERROR: loader order changed between passes (set shuffle=False / fixed order)")
            row = {"case": ids[i]}
            for c in conds:
                if c == "matched":
                    t = texts[i]
                elif c == "mean":
                    t = mean_text
                else:
                    t = texts[perms[int(c[-1])][i]]
                logits = model(img, t.to(dev))
                pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0]
                if pred.shape != gt.shape:
                    sys.exit(f"ERROR: prediction {pred.shape} vs label {gt.shape}")
                for r, d in zip(REGIONS, dice_per_region(pred, gt)):
                    row[f"{c}_{r}"] = d
                row[f"{c}_mean"] = float(np.mean([row[f"{c}_{r}"] for r in REGIONS]))
            rows.append(row)
            print(f"[{i + 1}/{n}] {ids[i]} matched={row['matched_mean']:.4f} mean={row['mean_mean']:.4f} "
                  f"shuf0={row['shuffled_s0_mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(a.out, "per_case.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def col(c, r="mean"):
        return np.array([x[f"{c}_{r}"] for x in rows])

    summ = {"n": n, "dice": {c: {r: float(col(c, r).mean()) for r in REGIONS + ("mean",)} for c in conds}}
    shuf = np.mean([col(f"shuffled_s{k}") for k in range(a.shuffle_seeds)], axis=0)
    summ["gain_vs_mean"] = paired_boot(col("matched") - col("mean"))
    summ["gain_vs_shuffled"] = paired_boot(col("matched") - shuf)
    summ["gain_by_region_vs_shuffled"] = {
        r: paired_boot(col("matched", r) - np.mean([col(f"shuffled_s{k}", r) for k in range(a.shuffle_seeds)], 0))
        for r in REGIONS}
    g, lo, hi = min(summ["gain_vs_mean"], summ["gain_vs_shuffled"], key=lambda x: x[0])
    if g < NEGLIGIBLE_GAIN and hi < NEGLIGIBLE_UPPER:
        v = "NEGLIGIBLE: the matched report adds < 0.5 Dice over a case-independent/mismatched report"
    elif g >= USES_GAIN and lo > 0:
        v = "USES_TEXT: the matched report improves Dice by >= 1 point"
    else:
        v = "INTERMEDIATE: small but non-zero text effect - check per-region gains and seed spread"
    summ["verdict"] = v
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)

    f = lambda x: f"{100 * x:.2f}"
    lines = ["# TextBraTS report-substitution test", "", f"**Verdict:** {v}", "",
             f"n = {n} test cases. Dice (%) per region; mean = average of TC/WT/ET.", "",
             "| condition | TC | WT | ET | mean |", "|---|---|---|---|---|"]
    for c in conds:
        d = summ["dice"][c]
        lines.append(f"| {c} | {f(d['TC'])} | {f(d['WT'])} | {f(d['ET'])} | {f(d['mean'])} |")
    for k in ("gain_vs_mean", "gain_vs_shuffled"):
        m, l, h = summ[k]
        lines.append(f"\n{k}: {f(m)} Dice points (95% CI {f(l)} to {f(h)})")
    lines.append("\nper-region gain vs shuffled: " + ", ".join(
        f"{r} {f(m)} [{f(l)}, {f(h)}]" for r, (m, l, h) in summ["gain_by_region_vs_shuffled"].items()))
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
