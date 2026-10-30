"""Side-word statistics of the LViT annotations per split, and agreement of one-sided reports with the
image half that holds the mask. Also cross-tabulates laterality words against side words (MosMed).
  python side_stats.py --lvit-dir $DATA/lvit_texts --qata-dir $DATA/qata/QaTa-COV19-v2 \
      --mosmed-masks $DATA/mosmed_1ch/masks
"""
import argparse, collections, os, re
import numpy as np, pandas as pd
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--lvit-dir", required=True, help="folder with the LViT *.xlsx text files")
ap.add_argument("--qata-dir", required=True, help="folder containing 'Train Set' and 'Test Set'")
ap.add_argument("--mosmed-masks", required=True, help="MosMedData+ mask folder (binary PNG)")
a = ap.parse_args()
L = a.lvit_dir


def norm(df):
    ic = "Image" if "Image" in df else df.columns[0]
    tc = "Description" if "Description" in df else [c for c in df.columns if c != ic][0]
    return list(zip(df[ic].astype(str).str.strip(), df[tc].astype(str)))


def qmask(split_dir):
    return lambda n: os.path.join(split_dir, "Ground-truths", n if n.startswith("mask_") else "mask_" + n)


mos = lambda n: os.path.join(a.mosmed_masks, n)
splits = [
    ("QaTa train+val", norm(pd.read_excel(f"{L}/Train_Val_text_for_Covid19.xlsx")), qmask(os.path.join(a.qata_dir, "Train Set"))),
    ("QaTa test", norm(pd.read_excel(f"{L}/Test_text_for_Covid19.xlsx")), qmask(os.path.join(a.qata_dir, "Test Set"))),
    ("MosMed train", norm(pd.read_excel(f"{L}/Train_text_MosMedData.xlsx")), mos),
    ("MosMed val", norm(pd.read_excel(f"{L}/Val_text_MosMedData.xlsx")), mos),
    ("MosMed test", norm(pd.read_excel(f"{L}/Test_text_MosMedData+.xlsx")), mos)]
LOC = re.compile(r"\b(all|upper|middle|lower)\s+(left|right)\s+lung", re.I)
for name, items, mpath in splits:
    side_cat, lat, combos, cross = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    img_left = {"left": [], "right": []}
    for img, t in items:
        sides = {s.lower() for s in re.findall(r"\b(left|right)\s+lung", t, re.I)}
        cat = "both" if len(sides) == 2 else (next(iter(sides)) + "-only" if sides else "no side word")
        lw = "Bilateral" if re.search(r"\bBilateral\b", t) else "Unilateral" if re.search(r"\bUnilateral\b", t) else "neither"
        side_cat[cat] += 1; lat[lw] += 1; cross[(lw, cat)] += 1
        combos[" + ".join(sorted(f"{e.lower()} {s.lower()}" for e, s in LOC.findall(t))) or "-"] += 1
        if len(sides) == 1 and os.path.exists(mpath(img)):
            m = np.array(Image.open(mpath(img)).convert("L")) > 127
            if m.any():
                img_left[next(iter(sides))].append(m[:, : m.shape[1] // 2].sum() / m.sum())
    print(f"\n== {name}: n={len(items)} | {dict(side_cat)} | {dict(lat)}")
    print("   laterality word x side words:", dict(cross))
    print("   top location combos:", combos.most_common(6))
    for s, v in img_left.items():
        if v:
            print(f"   '{s}-only' reports with mask: n={len(v)}, mask mostly on IMAGE-left in {np.mean(np.array(v) > 0.5):.0%}")
