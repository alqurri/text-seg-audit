"""Step 2 (TotalSegmentator env): lung lobes per scan -> work/lungs/<case>.nii.gz with labels
1 LUL, 2 LLL, 3 RUL, 4 RML, 5 RLL, in the CT's own voxel grid.

Laterality in the analysis comes from these masks, not from header conventions, so a wrong
orientation in VoxTell's input shows up as a failed lung control instead of silently flipping results.

  python lungs.py --work work [--fast] [--device gpu]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LOBES, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--fast", action="store_true", help="TotalSegmentator 3 mm model (enough for side assignment)")
    ap.add_argument("--device", default="gpu")
    a = ap.parse_args()

    import nibabel as nib
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    name2id = {v: k for k, v in class_map["total"].items()}
    remap = {name2id[n]: ours for ours, n in LOBES.items()}
    out_dir = os.path.join(a.work, "lungs")
    os.makedirs(out_dir, exist_ok=True)
    scans = read_jsonl(os.path.join(a.work, "scans.jsonl"))
    for i, s in enumerate(scans, 1):
        out = os.path.join(out_dir, f"{s['case']}.nii.gz")
        if os.path.exists(out):
            continue
        t = time.time()
        seg = totalsegmentator(s["image"], None, ml=True, fast=a.fast, roi_subset=list(LOBES.values()),
                               device=a.device, quiet=True)
        arr = np.asanyarray(seg.dataobj)
        lab = np.zeros(arr.shape, np.uint8)
        for ts_id, ours in remap.items():
            lab[arr == ts_id] = ours
        ref = nib.load(s["image"])
        if lab.shape != tuple(ref.shape):
            raise RuntimeError(f"{s['case']}: TotalSegmentator output {lab.shape} != CT {ref.shape}")
        nib.save(nib.Nifti1Image(lab, ref.affine), out + ".tmp.nii.gz")
        os.replace(out + ".tmp.nii.gz", out)
        present = sorted(int(v) for v in np.unique(lab) if v)
        print(f"[{i}/{len(scans)}] {s['case']} lobes={present} {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
