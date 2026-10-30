"""Do report side words agree with where the lesion is in the image?

For every test report naming exactly one side ("Unilateral ... left lung" etc.), compute the share of
mask pixels in the image's left half. Prints, per dataset, the image-side distribution for 'left' and
'right' reports. Consistent convention -> the two groups separate cleanly (radiological convention:
patient's left appears on image right). No separation -> the side words carry little usable signal.

  python $AUDIT/lat_check.py --folder $EXT/RecLMIS/datasets/Covid19/Test_Folder --covid
  python $AUDIT/lat_check.py --folder $EXT/RecLMIS/datasets/MosMedplus/Test_Folder
"""
import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--covid", action="store_true", help="QaTa: report keys are mask names")
    a = ap.parse_args()
    df = pd.read_excel(os.path.join(a.folder, "Test_text.xlsx"))
    col = "Description" if "Description" in df else df.columns[1]
    groups = {"left": [], "right": []}
    for img, t in zip(df.Image, df[col]):
        sides = set(re.findall(r"\b(left|right)\s+lung", str(t).lower()))
        if len(sides) != 1:
            continue
        m = cv2.imread(os.path.join(a.folder, "labelcol", img), 0)
        if m is None or not (m > 0).any():
            continue
        w = m.shape[1]
        frac_img_left = float((m[:, : w // 2] > 0).sum() / (m > 0).sum())
        groups[sides.pop()].append(frac_img_left)
    for s, v in groups.items():
        v = np.array(v)
        if len(v):
            print(f"'{s} lung' reports: n={len(v)}  mask share on IMAGE-left: median {np.median(v):.2f}  "
                  f"(>0.5 in {np.mean(v > 0.5):.0%})")
    l, r = np.array(groups["left"]), np.array(groups["right"])
    if len(l) and len(r):
        acc = (np.mean(l < 0.5) + np.mean(r > 0.5)) / 2
        print(f"side words predict image side (radiological convention) with balanced accuracy {acc:.2f} "
              f"(1.0 = perfectly consistent, 0.5 = no relation, 0.0 = consistent but opposite convention)")


if __name__ == "__main__":
    main()
