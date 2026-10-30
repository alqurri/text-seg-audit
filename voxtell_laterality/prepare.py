"""Step 1: parse ReXGroundingCT findings, build orig/swap/strip prompts, resolve and validate files.

  python prepare.py --rex-dir $D/rexgroundingct --ctrate-dir $D/ctrate --work work \
      --source dataset.json:val --source MICCAI_challenge_dataset.json:val
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (case_stem, finding_index_map, index_files, load_entries, load_nifti,  # noqa: E402
                    load_rex_mask, voxel_spacing, write_jsonl)
from swap import eligibility, strip_side, swap_ok, swap_side  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rex-dir", required=True)
    ap.add_argument("--ctrate-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--source", action="append", default=[], help="json_file:split (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask-hint", default="", help="substring a mask path must contain, if several match a case")
    ap.add_argument("--no-mask-check", action="store_true", help="skip loading masks (faster, less safe)")
    a = ap.parse_args()
    if not a.source:
        a.source = ["dataset.json:val", "MICCAI_challenge_dataset.json:val"]

    img_idx = index_files(a.ctrate_dir, (".nii.gz", ".nii"))
    mask_idx = index_files(a.rex_dir)

    train_names = None          # None = overlap unknown (recorded as null, never as "no overlap")
    try:
        train_names = {case_stem(e["name"]) for e in load_entries(os.path.join(a.rex_dir, "dataset.json"), "train")}
    except Exception as e:
        print(f"WARNING: could not read the ReXGroundingCT train list; training overlap recorded as unknown ({e})")

    scans, findings, seen = [], [], set()
    reasons = collections.Counter()
    for spec in a.source:
        jf, split = spec.split(":")
        entries = load_entries(os.path.join(a.rex_dir, jf), split)
        if a.limit:
            entries = entries[: a.limit]
        for e in entries:
            case = case_stem(e["name"])
            if case in seen:
                continue
            seen.add(case)
            imgs = img_idx.get(case, [])
            masks = [p for p in mask_idx.get(case, []) if p not in imgs]
            if len(masks) > 1 and a.mask_hint:
                masks = [p for p in masks if a.mask_hint in p]
            if len(masks) > 1:
                print(f"{case}: several candidate masks {masks} -> pass --mask-hint")
            if len(imgs) != 1 or len(masks) != 1:
                print(f"skip {case}: {len(imgs)} image file(s), {len(masks)} mask file(s)")
                reasons["missing_files"] += 1
                continue
            img, _ = load_nifti(imgs[0])  # header + lazy data
            shape = tuple(img.shape)
            fmap = finding_index_map(e["findings"])
            if not a.no_mask_check:
                m = load_rex_mask(masks[0], len(fmap), shape)
                nonempty = [bool((m[c] > 0).any()) for c in range(m.shape[0])]
            else:
                nonempty = [True] * len(fmap)
            scans.append({"case": case, "source": spec, "image": imgs[0], "mask": masks[0],
                          "shape": list(shape), "json_shape": e.get("shape"),
                          "spacing": voxel_spacing(img.affine).round(3).tolist(),
                          "n_findings": len(fmap), "in_rex_train": (case in train_names) if train_names is not None else None})
            cats = e.get("categories", {}) or {}
            ents = e.get("entity_counts", {}) or {}
            for key, ch in fmap.items():
                text = e["findings"][key]
                side, why = eligibility(text)
                row = {"case": case, "key": key, "ch": ch, "text": text,
                       "category": cats.get(key), "entities": ents.get(key),
                       "mask_nonempty": nonempty[ch], "side": side, "reason": why}
                if side and not nonempty[ch]:
                    row["reason"] = why = "empty_mask"
                    row["side"] = side = None
                if side:
                    row["swap"] = swap_side(text)
                    row["strip"] = strip_side(text)
                    if not swap_ok(text, row["swap"], side):
                        print(f"swap rejected: {text!r} -> {row['swap']!r}")
                        row["reason"] = why = "swap_failed"
                        row["side"] = side = None
                reasons[why] += 1
                findings.append(row)

    os.makedirs(a.work, exist_ok=True)
    write_jsonl(os.path.join(a.work, "scans.jsonl"), scans)
    write_jsonl(os.path.join(a.work, "findings.jsonl"), findings)

    elig = [f for f in findings if f["side"]]
    print(f"\nscans: {len(scans)} | findings: {len(findings)} | eligible (one side named): {len(elig)}")
    print("reasons:", dict(reasons))
    print("eligible by side:", dict(collections.Counter(f["side"] for f in elig)))
    print("eligible with a strip prompt:", sum(f["strip"] is not None for f in elig))
    ov = sum(bool(s["in_rex_train"]) for s in scans)
    unk = sum(s["in_rex_train"] is None for s in scans)
    print(f"ReXGroundingCT train-list overlap: {ov} scans" + (f" ({unk} unknown)" if unk else ""))
    if ov:
        print(f"WARNING: {ov} scans also appear in ReXGroundingCT train (possible VoxTell training exposure)")
    bad = [s["case"] for s in scans if s["json_shape"] and list(s["json_shape"]) != s["shape"]]
    if bad:
        print(f"NOTE: {len(bad)} scans whose json 'shape' differs from the NIfTI shape (e.g. {bad[:3]}); "
              "masks were validated against the NIfTI, so this is informational unless mask loading failed.")
    # intensity sanity on 3 scans: CT should reach about -1000 HU (air)
    for s in scans[:3]:
        _, arr = load_nifti(s["image"])
        lo, hi = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
        flag = "" if lo < -500 else "  <-- not HU? check RescaleSlope/Intercept (CT-RATE metadata)"
        print(f"intensity {s['case']}: p0.5={lo:.0f} p99.5={hi:.0f} spacing={s['spacing']}{flag}")
    print("\nexamples:")
    for f in elig[:8]:
        print(json.dumps({k: f[k] for k in ("text", "swap", "strip")}))


if __name__ == "__main__":
    main()
