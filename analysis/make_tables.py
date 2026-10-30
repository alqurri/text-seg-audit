"""Recompute the paper's numbers (Table 1, Fig. 1 slot effects, paraphrase controls) from results/.

  python analysis/make_tables.py --results results

CIs printed here are image-level bootstraps as stored by the runners; the paper reports the patient/volume-clustered
intervals from analysis/cluster_ci.py (same point estimates). Verdicts follow the paper: "uses text" if Delta >= 1 point with CI lower bound > 0; "equivalent" if the whole
95% CI lies within +-0.5 points; otherwise "intermediate".
"""
import argparse
import json
import os

import numpy as np


def load(r, *p):
    with open(os.path.join(r, *p)) as fh:
        return json.load(fh)


def verdict(m, lo, hi):
    if m >= 1 and lo > 0:
        return "uses text"
    if lo >= -0.5 and hi <= 0.5:
        return "equivalent"
    return "intermediate"


def fmt(t, nd=1):
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def tb_boot(agree_rows, key):
    rng = np.random.default_rng(0)
    if key == "shuffled":
        d = np.array([np.mean([x[f"shuffled_s{k}"]["gt_dice_delta"] for k in range(3)]) for x in agree_rows]) * 100
    else:
        d = np.array([x["mean"]["gt_dice_delta"] for x in agree_rows]) * 100
    bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    R = ap.parse_args().results
    rows = []

    # TextBraTS (dice stored as fractions; Delta from the prediction-agreement run for more precision)
    tb = load(R, "textbrats_kt", "summary.json")
    ag = load(R, "textbrats_agree", "agree.json")["rows"]
    shuf = np.mean([tb["dice"][f"shuffled_s{k}"]["mean"] for k in range(3)])
    dc, ds = tb_boot(ag, "constant"), tb_boot(ag, "shuffled")
    rows.append(("TextBraTS / BraTS MRI", tb["n"], 100 * tb["dice"]["matched"]["mean"], 100 * tb["dice"]["mean"]["mean"],
                 100 * shuf, dc, ds, None, 3))

    # RecLMIS (fractions)
    para_r = {"qata": load(R, "para_lenmatch", "all_lenmatched", "summary.json")["slots"]["para"],
              "mosmed": load(R, "para2_reclmis_mosmed", "summary.json")["slots"]["para"]}
    for ds_name, tag in (("qata", "QaTa X-ray"), ("mosmed", "MosMed CT")):
        s = load(R, f"reclmis_{ds_name}", "summary.json")
        sh = np.mean([s["dice"][f"shuffled_s{k}"] for k in range(3)])
        g = lambda k: tuple(100 * x for x in s[k])
        p = para_r[ds_name]
        rows.append((f"RecLMIS / {tag}", s["n"], 100 * s["dice"]["matched"], 100 * s["dice"]["constant"], 100 * sh,
                     g("gain_vs_constant"), g("gain_vs_shuffled"),
                     (100 * p["effect"], 100 * p["ci95"][0], 100 * p["ci95"][1]), 1))

    # Ariadne's Thread (points)
    for ds_name, tag in (("qata", "QaTa X-ray"), ("mosmed", "MosMed CT")):
        s = load(R, f"ariadne_{ds_name}", "summary.json")
        p = load(R, f"para2_ariadne_{ds_name}", "summary.json")["delta"]["slot_para"]
        d = lambda k: (s["delta"][k]["mean"], s["delta"][k]["lo"], s["delta"][k]["hi"])
        rows.append((f"Ariadne / {tag}", s["n"], s["dice"]["matched"], s["dice"]["constant"], s["dice"]["shuffled_mean"],
                     d("constant"), d("shuffled_mean"), (p["mean"], p["lo"], p["hi"]), 1))

    print("Table 1  (Dice %, Delta in Dice points with 95% paired bootstrap CI)")
    print(f"{'setting':24s} {'n':>5s} {'match':>6s} {'const':>6s} {'shuf':>6s}  {'D const':22s} {'D shuf':22s} {'D para':18s} verdict")
    for name, n, m, c, s, dc, dsh, dp, nd in rows:
        print(f"{name:24s} {n:5d} {m:6.2f} {c:6.2f} {s:6.2f}  {fmt(dc, nd):22s} {fmt(dsh, nd):22s} "
              f"{fmt(dp) if dp else '-':18s} {verdict(*dc)} / {verdict(*dsh)}")

    print("\nFig. 1  single-slot edits (Dice points lost, 95% CI, n)")
    for model in ("RecLMIS", "Ariadne"):
        for ds_name in ("qata", "mosmed"):
            out = []
            for slot in ("side", "extent", "count", "lat"):
                if model == "RecLMIS":
                    s = load(R, f"slots_{ds_name}", "summary.json")["slots"][slot]
                    t, n = (100 * s["effect"], 100 * s["ci95"][0], 100 * s["ci95"][1]), s["n"]
                else:
                    s = load(R, f"ariadne_{ds_name}", "summary.json")["delta"][f"slot_{slot}"]
                    t, n = (s["mean"], s["lo"], s["hi"]), s["n"]
                out.append(f"{slot} {fmt(t, 2)} n={n}")
            print(f"  {model:8s} {ds_name:7s} " + " | ".join(out))

    u = load(R, "para_untrunc_qata", "summary.json")["slots"]["para_untrunc"]
    print(f"\nRecLMIS/QaTa paraphrase on reports untruncated before and after rewriting: "
          f"{fmt((100 * u['effect'], 100 * u['ci95'][0], 100 * u['ci95'][1]))} n={u['n']}")
    print("Single paraphrase rules (Dice points):")
    for j in range(6):
        r = load(R, "pararules_reclmis_qata", f"rule{j}", "summary.json")["slots"]["para"]
        a = load(R, "pararules_ariadne_qata", f"rule{j}", "summary.json")["delta"]["slot_para"]
        print(f"  rule {j}: RecLMIS {100 * r['effect']:5.2f} (n={r['n']})   Ariadne {a['mean']:5.2f} (n={a['n']})")
    for ds_name in ("qata", "mosmed"):
        nm = load(R, f"nomask_{ds_name}", "nomask.json")
        print(f"RecLMIS zero-mask check {ds_name}: identical masks {nm['identical_masks']}/{nm['n']}, "
              f"max |output diff| {nm['max_abs_output_diff']}, {nm['seconds_per_pass']:.1f} s per pass")


if __name__ == "__main__":
    main()
