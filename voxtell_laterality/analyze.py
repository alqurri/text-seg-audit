"""Step 4: metrics and the verdict.

Primary metric - Swap Failure Rate (SFR): among findings the ORIGINAL prompt actually hits, the share
where the laterality-SWAPPED prompt still segments >= 50% of what the original prompt hit on the
original (now wrong-side) lesion. SFR ~ 0 means the model reads location words; SFR high means it ignores them.

Gates (verdict ABORT if failed - these indicate an I/O or orientation bug, not a finding):
  lung control: "left lung"/"right lung" land on the right side in >= 95% of scans
  alignment:    ReXGroundingCT masks sit inside the thorax ROI from the lobe masks (median >= 0.85)

Decision fixed before the run (point estimate and cluster-bootstrap 95% CI over scans):
  FOLLOWS_LATERALITY  if SFR <= 0.10 and CI upper <= 0.15
  IGNORES_LATERALITY  if SFR >= 0.30
  else AMBIGUOUS (read the per-category breakdown)
  UNDERPOWERED if fewer than 40 hit findings

  python analyze.py --work work
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (LEFT_LABELS, RIGHT_LABELS, dice, load_nifti, load_packed, load_rex_mask,  # noqa: E402
                    read_jsonl, side_fractions, side_map, voxel_spacing)
from swap import opposite  # noqa: E402

CFG = dict(hit=0.10, stale=0.50, suppress=0.10, consistent=0.80, in_roi=0.80,
           ctrl_side=0.90, ctrl_dice=0.70, gate_ctrl=0.95, gate_align=0.85,
           follow_point=0.10, follow_upper=0.15, ignore_point=0.30, min_hits=40, n_boot=2000, seed=0)


def entity_sides(chmask, side, roi):
    """Per-entity dominant side for one finding channel (entity ids are the voxel values)."""
    out = []
    for v in np.unique(chmask):
        if v == 0:
            continue
        sf = side_fractions(chmask == v, side, roi)
        if sf["n"] and not np.isnan(sf["left"]):
            out.append((int(v), "left" if sf["left"] >= 0.5 else "right"))
    return out


def analyze_scan(s, frows, work, cfg):
    _, lobes = load_nifti(os.path.join(work, "lungs", f"{s['case']}.nii.gz"))
    spacing = voxel_spacing(load_nifti(s["image"])[0].affine)
    side, roi = side_map(lobes, spacing)
    preds = load_packed(os.path.join(work, "preds", f"{s['case']}.npz"))
    gt = load_rex_mask(s["mask"], s["n_findings"], s["shape"])

    lungL, lungR = np.isin(lobes, LEFT_LABELS), np.isin(lobes, RIGHT_LABELS)
    ctrl = {}
    for k, want, ref in (("ctrl_left", "left", lungL), ("ctrl_right", "right", lungR)):
        sf = side_fractions(preds[k], side, roi)
        ctrl[k] = {"side_frac": sf[want], "dice": dice(preds[k], ref),
                   "pass": bool(sf["n"] and sf[want] >= cfg["ctrl_side"] and dice(preds[k], ref) >= cfg["ctrl_dice"])}
    scan_row = {"case": s["case"], "ctrl_pass": ctrl["ctrl_left"]["pass"] and ctrl["ctrl_right"]["pass"],
                **{f"{k}_{m}": v[m] for k, v in ctrl.items() for m in ("side_frac", "dice")}}

    # entity-level side for every finding in the scan (used for counterfactual targets)
    cats = {f["ch"]: f.get("category") for f in frows}
    ent = {ch: entity_sides(gt[ch], side, roi) for ch in range(gt.shape[0])}

    rows, align = [], []
    for f in frows:
        g = gt[f["ch"]] > 0
        gsf = side_fractions(g, side, roi)
        if gsf["n"]:
            align.append(gsf["in_roi"])
        if not f["side"]:
            continue
        r = {"case": s["case"], "key": f["key"], "text": f["text"], "swap": f["swap"], "category": f.get("category"),
             "side": f["side"], "swap_kind": ("anatomy_word" if any(w in f["text"].lower() for w in ("middle lobe", "lingula", "rml")) else "side_word_only"), "has_lobe": any(w in f["text"].lower() for w in ("lobe", "lingula", "rul", "lll", "lul", "rll", "rml")),
             "gt_vox": gsf["n"], "gt_side_frac": gsf[f["side"]], "gt_in_roi": gsf["in_roi"]}
        if not gsf["n"] or gsf["in_roi"] < cfg["in_roi"] or gsf[f["side"]] < cfg["consistent"]:
            r["excluded"] = "text_mask_side_mismatch" if gsf["n"] and gsf["in_roi"] >= cfg["in_roi"] else "outside_thorax"
            rows.append(r)
            continue
        k = f"f{f['key']}"
        po, ps = preds[f"{k}_orig"], preds[f"{k}_swap"]
        hit_o = (po & g).sum()
        r.update(dice_orig=dice(po, g), hit_frac_orig=hit_o / g.sum(), pred_vox_orig=int(po.sum()),
                 pred_vox_swap=int(ps.sum()))
        r["hit"] = bool(r["hit_frac_orig"] >= cfg["hit"])
        r["stale_rel"] = float((ps & g).sum() / hit_o) if hit_o else np.nan
        r["swap_fail"] = bool(r["hit"] and r["stale_rel"] >= cfg["stale"])
        r["suppressed"] = bool(ps.sum() < cfg["suppress"] * max(po.sum(), 1))
        ssf = side_fractions(ps, side, roi)
        r["swap_side_frac_target"] = ssf[opposite(f["side"])]
        r["dice_swap_vs_orig"] = dice(ps, po)
        # counterfactual target: same-category entities on the swapped side, from any finding in the scan
        cf = np.zeros_like(g)
        for ch, es in ent.items():
            if cats.get(ch) is not None and cats.get(ch) == f.get("category"):
                for v, sd in es:
                    if sd == opposite(f["side"]):
                        cf |= gt[ch] == v
        r["cf_exists"] = bool(cf.any())
        r["dice_swap_vs_cf"] = dice(ps, cf) if cf.any() else np.nan
        if f"{k}_strip" in preds:
            pst = preds[f"{k}_strip"]
            r["hit_frac_strip"] = (pst & g).sum() / g.sum()
            r["dice_strip"] = dice(pst, g)
        rows.append(r)
    scan_row["align_median"] = float(np.median(align)) if align else np.nan
    return scan_row, rows


def boot_ci(per_case_values, stat, n, seed):
    """Cluster bootstrap over scans. per_case_values: list of arrays (one per scan)."""
    rng = np.random.default_rng(seed)
    cases = [v for v in per_case_values if len(v)]
    if not cases:
        return np.nan, (np.nan, np.nan)
    point = stat(np.concatenate(cases))
    bs = [stat(np.concatenate([cases[i] for i in rng.integers(0, len(cases), len(cases))])) for _ in range(n)]
    return float(point), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def summarize(rows, scans_rows, cfg):
    ok = [r for r in rows if "excluded" not in r]
    hits = [r for r in ok if r["hit"]]
    by_case = collections.defaultdict(list)
    for r in hits:
        by_case[r["case"]].append(r)
    sfr = boot_ci([np.array([x["swap_fail"] for x in v], float) for v in by_case.values()], np.mean,
                  cfg["n_boot"], cfg["seed"])
    stale = boot_ci([np.array([x["stale_rel"] for x in v], float) for v in by_case.values()], np.nanmedian,
                    cfg["n_boot"], cfg["seed"])
    ctrl_rate = np.mean([s["ctrl_pass"] for s in scans_rows]) if scans_rows else np.nan
    align = np.nanmedian([s["align_median"] for s in scans_rows]) if scans_rows else np.nan

    def rate(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return (float(np.mean(v)), len(v)) if v else (np.nan, 0)

    strip_sel = [r for r in hits if "hit_frac_strip" in r]
    out = {
        "n_scans": len(scans_rows), "n_findings_eligible": len(rows), "n_excluded": len(rows) - len(ok),
        "excluded_reasons": dict(collections.Counter(r["excluded"] for r in rows if "excluded" in r)),
        "n_scored": len(ok), "n_hit": len(hits), "hit_rate_orig": rate(ok, "hit")[0],
        "dice_orig_mean": rate(ok, "dice_orig")[0],
        "gate_lung_control_pass_rate": ctrl_rate, "gate_alignment_median_in_roi": align,
        "SFR": sfr[0], "SFR_CI95": sfr[1], "stale_rel_median": stale[0], "stale_rel_median_CI95": stale[1],
        "swap_suppressed_rate": rate(hits, "suppressed")[0],
        "swap_correct_side_mean_(non-empty)": float(np.nanmean([r["swap_side_frac_target"] for r in hits if not r["suppressed"]])) if hits else np.nan,
        "dice_swap_vs_cf_mean_(cf exists)": float(np.nanmean([r["dice_swap_vs_cf"] for r in hits if r["cf_exists"]])) if any(r["cf_exists"] for r in hits) else np.nan,
        "strip_hit_rate_(among orig hits)": float(np.mean([r["hit_frac_strip"] >= cfg["hit"] for r in strip_sel])) if strip_sel else np.nan,
        "by_side": {sd: rate([r for r in hits if r["side"] == sd], "swap_fail") for sd in ("left", "right")},
        "by_swap_kind": {k: rate([r for r in hits if r["swap_kind"] == k], "swap_fail") for k in ("side_word_only", "anatomy_word")},
        "by_has_lobe": {str(h): rate([r for r in hits if r["has_lobe"] == h], "swap_fail") for h in (True, False)},
        "by_category": {str(c): rate([r for r in hits if r["category"] == c], "swap_fail")
                        for c in sorted({str(r["category"]) for r in hits})},
    }
    # verdict
    if ctrl_rate < cfg["gate_ctrl"]:
        v = "ABORT: lung control failed - check orientation / reader before interpreting anything"
    elif not align >= cfg["gate_align"]:
        v = "ABORT: ReXGroundingCT masks do not sit in the thorax - mask/CT alignment problem"
    elif len(hits) < cfg["min_hits"]:
        v = f"UNDERPOWERED: only {len(hits)} hit findings (need >= {cfg['min_hits']})"
    elif sfr[0] <= cfg["follow_point"] and sfr[1][1] <= cfg["follow_upper"]:
        v = "FOLLOWS_LATERALITY: swapped prompts rarely keep the original lesion"
    elif sfr[0] >= cfg["ignore_point"]:
        v = "IGNORES_LATERALITY: swapped prompts often keep the original lesion"
    else:
        v = "AMBIGUOUS: read by_category / by_has_lobe before deciding"
    out["verdict"] = v
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5:
        return np.nan, np.nan
    rho, p = spearmanr(x[m], y[m])
    return float(rho), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    cfg = CFG
    scans = read_jsonl(os.path.join(a.work, "scans.jsonl"))
    findings = read_jsonl(os.path.join(a.work, "findings.jsonl"))
    by_case = collections.defaultdict(list)
    for f in findings:
        by_case[f["case"]].append(f)

    scan_rows, rows, missing = [], [], []
    for s in scans:
        need = [os.path.join(a.work, "lungs", f"{s['case']}.nii.gz"), os.path.join(a.work, "preds", f"{s['case']}.npz")]
        if not all(os.path.exists(p) for p in need):
            missing.append(s["case"])
            continue
        sr, rr = analyze_scan(s, by_case[s["case"]], a.work, cfg)
        scan_rows.append(sr)
        rows += rr

    cos = {}
    cos_path = os.path.join(a.work, "embed_cos.jsonl")
    if os.path.exists(cos_path):
        cos = {(c["case"], c["key"]): c for c in read_jsonl(cos_path)}
    for r in rows:
        c = cos.get((r["case"], r["key"]), {})
        r["cos_swap"], r["cos_strip"] = c.get("cos_swap", np.nan), c.get("cos_strip", np.nan)

    summ = summarize(rows, scan_rows, cfg)
    hits = [r for r in rows if "excluded" not in r and r["hit"]]
    summ["cos_swap_median"] = float(np.nanmedian([r["cos_swap"] for r in hits])) if hits else np.nan
    summ["spearman_cos_swap_vs_stale"] = spearman([r["cos_swap"] for r in hits], [r["stale_rel"] for r in hits])
    summ["missing_scans"] = missing
    summ["config"] = cfg

    res = os.path.join(a.work, "results")
    os.makedirs(res, exist_ok=True)
    for name, data in (("per_finding.csv", rows), ("per_scan.csv", scan_rows)):
        if data:
            keys = sorted({k for d in data for k in d})
            with open(os.path.join(res, name), "w", newline="") as fh:
                w = csv.DictWriter(fh, keys)
                w.writeheader()
                w.writerows(data)
    json.dump(summ, open(os.path.join(res, "summary.json"), "w"), indent=2, default=float)
    write_report(os.path.join(res, "report.md"), summ, hits)
    print(json.dumps({k: summ[k] for k in ("verdict", "SFR", "SFR_CI95", "n_hit", "gate_lung_control_pass_rate",
                                           "gate_alignment_median_in_roi", "dice_orig_mean", "cos_swap_median")},
                     indent=2, default=float))
    if missing:
        print(f"{len(missing)} scans skipped (no lungs/preds yet)")


def write_report(path, s, hits):
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))
    lines = [
        "# Laterality audit - VoxTell on ReXGroundingCT", "", f"**Verdict:** {s['verdict']}", "",
        "| quantity | value |", "|---|---|",
        f"| scans / eligible findings / scored / hit | {s['n_scans']} / {s['n_findings_eligible']} / {s['n_scored']} / {s['n_hit']} |",
        f"| gate: lung control pass rate | {f(s['gate_lung_control_pass_rate'])} |",
        f"| gate: median GT fraction in thorax ROI | {f(s['gate_alignment_median_in_roi'])} |",
        f"| original-prompt Dice (mean, scored findings) | {f(s['dice_orig_mean'])} |",
        f"| original-prompt hit rate | {f(s['hit_rate_orig'])} |",
        f"| **Swap Failure Rate** | **{f(s['SFR'])}** (95% CI {f(s['SFR_CI95'][0])}-{f(s['SFR_CI95'][1])}) |",
        f"| median stale overlap (swap / orig) | {f(s['stale_rel_median'])} |",
        f"| swap suppressed (swap volume < 10% of original prediction) | {f(s['swap_suppressed_rate'])} |",
        f"| share of swap voxels inside the thorax ROI on the swapped side (non-suppressed) | {f(s['swap_correct_side_mean_(non-empty)'])} |",
        f"| Dice(swap, contralateral same-category lesion) | {f(s['dice_swap_vs_cf_mean_(cf exists)'])} |",
        f"| side-stripped prompt still hits | {f(s['strip_hit_rate_(among orig hits)'])} |",
        f"| median cosine(orig, swap) text embedding | {f(s['cos_swap_median'])} |",
        f"| Spearman(cosine, stale overlap) | {f(s['spearman_cos_swap_vs_stale'][0])} (p={f(s['spearman_cos_swap_vs_stale'][1])}) |",
        "", "SFR by side: " + json.dumps(s["by_side"], default=float),
        "", "SFR by swap kind (side_word_only = only right/left or RUL-type words change): " + json.dumps(s["by_swap_kind"], default=float),
        "", "SFR by lobe named: " + json.dumps(s["by_has_lobe"], default=float),
        "", "SFR by category: " + json.dumps(s["by_category"], default=float),
        "", f"Excluded: {json.dumps(s['excluded_reasons'])}", "", "## Worst swap failures", "",
        "| case | text | swap | stale | Dice orig |", "|---|---|---|---|---|",
    ]
    for r in sorted(hits, key=lambda r: -np.nan_to_num(r["stale_rel"]))[:15]:
        lines.append(f"| {r['case']} | {r['text'][:80]} | {r['swap'][:80]} | {f(r['stale_rel'])} | {f(r['dice_orig'])} |")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
