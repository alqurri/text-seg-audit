"""Patient/volume-clustered paired bootstrap CIs for the 2D datasets, from saved per-image results.

QaTa-COV19 file names of BIMCV origin carry a subject id ("sub-S08047_ses-..."); MosMedData+ slice names carry a
CT-volume id ("Morozov_study_0297_12.png" -> "Morozov_study_0297"). Images without a recognisable id form their own
cluster (so QaTa clusters are a lower bound on correlation, not a patient count). Resampling clusters instead of
images accounts for correlation between images of one patient/volume. 5,000 resamples, seed 0. Delta in Dice points.

  python analysis/cluster_ci.py --results results [--json out.json]
"""
import argparse
import collections
import csv
import json
import os
import re

import numpy as np


def cluster_of(name, dataset):
    n = os.path.basename(str(name))
    if dataset == "qata":
        m = re.match(r"(?:mask_)?(sub-[A-Za-z0-9]+)_", n)
        return m.group(1) if m else n
    return re.sub(r"_\d+\.png$", "", n)


def boot(diffs, clusters, n=5000, seed=0):
    by = collections.defaultdict(list)
    for d, c in zip(diffs, clusters):
        by[c].append(d)
    keys = list(by)
    sums = np.array([np.sum(by[k]) for k in keys])
    cnts = np.array([len(by[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (n, len(keys)))
    bs = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"mean": float(np.mean(diffs)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n": len(diffs), "clusters": len(keys)}


def verdict(r, margin=0.5, use=1.0):
    if r["mean"] >= use and r["lo"] > 0:
        return "uses text"
    if r["lo"] >= -margin and r["hi"] <= margin:
        return "equivalent"
    return "intermediate"


def load(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


def paired(base, cond, ds, scale):
    keys = [k for k in cond if k in base]
    d = [scale * (base[k] - cond[k]) for k in keys]
    return boot(d, [cluster_of(k, ds) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--json")
    a = ap.parse_args()
    R = a.results
    out = {}
    for ds in ("qata", "mosmed"):
        res = {}
        # RecLMIS substitution (per-image CSV, fractions)
        rows = list(csv.DictReader(open(os.path.join(R, f"reclmis_{ds}", "per_image.csv"))))
        col = lambda k: {r["image"]: float(r[k]) for r in rows}
        m = col("matched")
        shuf = {k: np.mean([col(f"shuffled_s{s}")[k] for s in range(3)]) for k in m}
        res["RecLMIS constant"] = paired(m, col("constant"), ds, 100)
        res["RecLMIS shuffled"] = paired(m, shuf, ds, 100)
        # RecLMIS slots and paraphrase (per_image.json, fractions)
        sl = load(R, f"slots_{ds}", "per_image.json")
        for s in ("side", "extent", "count", "lat"):
            res[f"RecLMIS slot {s}"] = paired(sl["matched"], sl[s], ds, 100)
        pr = load(R, "para_lenmatch", "all_lenmatched", "per_image.json") if ds == "qata" else \
            load(R, "para2_reclmis_mosmed", "per_image.json")
        res["RecLMIS paraphrase"] = paired(pr["matched"], pr["para"], ds, 100)
        if ds == "qata":
            pu = load(R, "para_untrunc_qata", "per_image.json")
            res["RecLMIS paraphrase (untruncated)"] = paired(pu["matched"], pu["para_untrunc"], ds, 100)
        # Ariadne's Thread (per_image.json, fractions in [0,1])
        pi = load(R, f"ariadne_{ds}", "per_image.json")
        mt = pi["matched"]
        res["Ariadne constant"] = paired(mt, pi["constant"], ds, 100)
        res["Ariadne shuffled"] = paired(mt, pi["shuffled_mean"], ds, 100)
        for s in ("side", "extent", "count", "lat"):
            sel = sl[s]                                   # identical edits (same rules and seed); score affected only
            res[f"Ariadne slot {s}"] = paired(mt, {k: v for k, v in pi[f"slot_{s}"].items() if k in sel}, ds, 100)
        pp = load(R, f"para2_ariadne_{ds}", "per_image.json")
        res["Ariadne paraphrase"] = paired(mt, pp["slot_para"], ds, 100)
        names = list(m)
        c = collections.Counter(cluster_of(n, ds) for n in names)
        print(f"\n== {ds}: {len(names)} images in {len(c)} clusters (largest {max(c.values())}, "
              f"singletons {sum(v == 1 for v in c.values())})")
        for k, r in res.items():
            print(f"  {k:34s} {r['mean']:6.2f} [{r['lo']:6.2f}, {r['hi']:6.2f}]  n={r['n']:4d} clusters={r['clusters']:3d}  {verdict(r)}")
        out[ds] = res
    print("\nThreshold sensitivity (verdicts that change with margin 0.25/1.0 or use threshold 0.5/2.0):")
    for ds, res in out.items():
        for k, r in res.items():
            base = verdict(r)
            alt = {f"margin {m}, use {u}": verdict(r, m, u) for m in (0.25, 1.0) for u in (0.5, 2.0)}
            flips = {x: y for x, y in alt.items() if y != base}
            if flips:
                print(f"  {ds:6s} {k:34s} {base:12s} -> {flips}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
