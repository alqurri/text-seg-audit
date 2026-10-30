"""VoxTell laterality audit bookkeeping from voxtell_laterality/analyze.py output: funnel, failures with
exact (Clopper-Pearson) intervals, breakdown of swap outcomes, overlap with the ReXGroundingCT training
list, and scan sources.   python voxtell_summary.py --work $OUT/voxtell_work"""
import argparse, collections, csv, json, os
import numpy as np
from scipy.stats import beta

ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); a = ap.parse_args()
R = a.work
rows = list(csv.DictReader(open(f"{R}/results/per_finding.csv")))
S = json.load(open(f"{R}/results/summary.json"))
B = lambda x: str(x).strip() == "True"
F = lambda x: float(x) if x not in ("", "nan", None) else np.nan
cp = lambda k, n: (beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
print("scans:", S.get("n_scans"), "| one-sided findings:", len(rows), "| excluded:", S.get("excluded_reasons"))
print("scan-cluster bootstrap SFR:", S.get("SFR"), S.get("SFR_CI95"))
ok = [r for r in rows if not r.get("excluded")]
hits = [r for r in ok if B(r["hit"])]
dsc = lambda sel: np.nanmean([F(r["dice_orig"]) for r in sel])
print(f"mean original-prompt Dice: {dsc(ok):.3f} over scored findings, {dsc([r for r in ok if B(r['hit'])]):.3f} over hits")
print(f"scored {len(ok)}; hit (orig prompt covers >=10% of GT lesion) {len(hits)}; scans with >=1 hit {len({r['case'] for r in hits})}")
for name, sub in (("all", hits), ("side_word_only", [r for r in hits if r["swap_kind"] == "side_word_only"]),
                  ("anatomy_word", [r for r in hits if r["swap_kind"] == "anatomy_word"])):
    k = sum(B(r["swap_fail"]) for r in sub); lo, hi = cp(k, len(sub))
    print(f"  {name}: fail {k}/{len(sub)}  Clopper-Pearson 95% [{100 * lo:.2f}, {100 * hi:.2f}]%")
sup = [r for r in hits if B(r["suppressed"])]; ns = [r for r in hits if not B(r["suppressed"])]
print(f"suppressed = swap volume < 10% of the original prediction: {len(sup)} "
      f"({sum(int(r['pred_vox_swap']) == 0 for r in sup)} completely empty)")
cfr = [r for r in hits if B(r["cf_exists"])]
dcf = np.array([F(r["dice_swap_vs_cf"]) for r in cfr])
print(f"same-category lesion on the named side exists for {len(cfr)} hits; swap overlaps it (Dice > 0) in "
      f"{int((dcf > 0).sum())}, mean Dice {np.nanmean(dcf):.3f}")
print(f"suppressed {len(sup)}: with a same-category contralateral lesion {sum(B(r['cf_exists']) for r in sup)}, "
      f"without {sum(not B(r['cf_exists']) for r in sup)}")
tf = np.array([F(r["swap_side_frac_target"]) for r in ns])
fails = sum(B(r["swap_fail"]) for r in ns)
print(f"non-suppressed {len(ns)}: mean share of swap voxels inside the thorax ROI on the swapped side {np.nanmean(tf):.3f}; >=80% on swapped side "
      f"{int(np.nansum(tf >= 0.8))}, failures {fails}, other {int(np.nansum(tf < 0.8)) - fails}; "
      f"contralateral lesion exists for {sum(B(r['cf_exists']) for r in ns)}")
mf = next((p for p in (f"{R}/scans.jsonl", f"{R}/scans_manifest.jsonl") if os.path.exists(p)), None)
if mf is None:
    raise SystemExit("(no scans.jsonl / scans_manifest.jsonl: run on the work folder written by voxtell_laterality/prepare.py "
                     "for the training-list overlap and scan sources)")
scans = [json.loads(l) for l in open(mf) if l.strip()]
print("scans in ReXGroundingCT train list:", sum(bool(s.get("in_rex_train")) for s in scans),
      "| unknown:", sum(s.get("in_rex_train") is None for s in scans),
      "| sources:", dict(collections.Counter(s.get("source") for s in scans)))
