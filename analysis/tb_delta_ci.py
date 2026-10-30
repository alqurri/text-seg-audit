"""TextBraTS text effect with more precision: paired bootstrap of the per-case change in ground-truth
Dice (points), from tb_agree.py output.  python tb_delta_ci.py --agree $OUT/textbrats_agree/agree.json"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--agree", required=True); a = ap.parse_args()
rows = json.load(open(a.agree))["rows"]
rng = np.random.default_rng(0)


def boot(d):
    d = np.asarray(d) * 100
    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), np.abs(d).max()


mean = [r["mean"]["gt_dice_delta"] for r in rows]
shuf = [np.mean([r[f"shuffled_s{k}"]["gt_dice_delta"] for k in range(3)]) for r in rows]
for name, d in (("constant", mean), ("shuffled", shuf)):
    print(name, "delta %.4f [%.4f, %.4f]  max|per-case| %.4f points" % boot(d))
