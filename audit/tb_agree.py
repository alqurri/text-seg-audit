"""Prediction-level invariance for TextBraTS: do masks change with the text, beyond rerun noise?"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shuffle_test import build_args, load_components, derangement, dice_per_region, REGIONS

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--json_list", default="Test.json")
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
get_loader, make_model = load_components(a); args = build_args(a)
dev = torch.device("cuda")
model = make_model(args)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd)
mk, unx = model.load_state_dict(sd, strict=False); print("missing", len(mk), "unexpected", len(unx))
if mk or unx:
    sys.exit("ERROR: checkpoint does not match the model exactly")
model.to(dev).eval(); loader = get_loader(args)
texts = []
for i, b in enumerate(loader):
    if a.limit and i >= a.limit: break
    texts.append(b["text_feature"].float().clone())
n = len(texts); mean_text = torch.stack(texts).mean(0)
rng = np.random.default_rng(0); perms = [derangement(n, rng) for _ in range(3)]
conds = ["matched_repeat", "mean", "shuffled_s0", "shuffled_s1", "shuffled_s2"]
def mask_dice(p, q):
    s = p.sum() + q.sum(); return 1.0 if s == 0 else float(2 * (p & q).sum() / s)
rows = []
with torch.no_grad():
    for i, b in enumerate(loader):
        if i >= n: break
        img = b["image"].to(dev); gt = b["label"].numpy()[0].astype(bool)
        p_ref = torch.sigmoid(model(img, texts[i].to(dev)))[0]; m_ref = (p_ref > 0.5)
        mr = m_ref.cpu().numpy(); d_ref = np.mean(dice_per_region(mr, gt))
        row = {"case": i}
        for c in conds:
            t = texts[i] if c == "matched_repeat" else mean_text if c == "mean" else texts[perms[int(c[-1])][i]]
            p = torch.sigmoid(model(img, t.to(dev)))[0]; m = (p > 0.5); mm = m.cpu().numpy()
            diff = (m != m_ref)
            row[c] = {"identical": bool(not diff.any()),
                      "flip_pct_all": float(diff.float().mean() * 100),
                      "flip_pct_of_fg": float(diff.sum() / max(int(m_ref.sum()), 1) * 100),
                      "pred_dice_min_region": min(mask_dice(mm[r], mr[r]) for r in range(3)),
                      "max_abs_prob": float((p - p_ref).abs().max()),
                      "mean_abs_prob": float((p - p_ref).abs().mean()),
                      "gt_dice_delta": float(d_ref - np.mean(dice_per_region(mm, gt)))}
        rows.append(row)
        print(i + 1, n, {c: (row[c]["identical"], round(row[c]["flip_pct_of_fg"], 4)) for c in conds}, flush=True)
summ = {}
for c in conds:
    v = lambda k: np.array([r[c][k] for r in rows], float)
    summ[c] = {"frac_identical_masks": float(v("identical").mean()),
               "flip_pct_all_mean": float(v("flip_pct_all").mean()), "flip_pct_all_max": float(v("flip_pct_all").max()),
               "flip_pct_of_fg_mean": float(v("flip_pct_of_fg").mean()), "flip_pct_of_fg_max": float(v("flip_pct_of_fg").max()),
               "pred_dice_min": float(v("pred_dice_min_region").min()), "pred_dice_mean": float(v("pred_dice_min_region").mean()),
               "max_abs_prob_max": float(v("max_abs_prob").max()), "mean_abs_prob_mean": float(v("mean_abs_prob").mean()),
               "gt_dice_delta_pts_mean": float(v("gt_dice_delta").mean() * 100),
               "gt_dice_delta_pts_maxabs": float(np.abs(v("gt_dice_delta")).max() * 100)}
json.dump({"n": n, "summary": summ, "rows": rows}, open(os.path.join(a.out, "agree.json"), "w"), indent=1)
for c in conds: print(c, json.dumps({k: (round(x, 6) if isinstance(x, float) else x) for k, x in summ[c].items()}))
