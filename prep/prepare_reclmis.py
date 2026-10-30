"""Lay out the QaTa-COV19 and MosMedData+ test sets the way RecLMIS' (LViT-style) loader expects:
  <RecLMIS>/datasets/Covid19/Test_Folder/{img,labelcol,Test_text.xlsx}
  <RecLMIS>/datasets/MosMedplus/Test_Folder/{img,labelcol,Test_text.xlsx}
Images/masks are symlinked (originals untouched); report columns are normalised to Image/Description.

  python prepare_reclmis.py --reclmis $EXT/RecLMIS --qata-dir $DATA/qata/QaTa-COV19-v2 \
      --mosmed-dir $DATA/mosmed --lvit-dir $DATA/lvit_texts
"""
import argparse, os
import pandas as pd


def norm(df):
    ic = "Image" if "Image" in df else df.columns[0]
    tc = "Description" if "Description" in df else [c for c in df.columns if c != ic][0]
    return pd.DataFrame({"Image": df[ic].astype(str).str.strip(), "Description": df[tc].astype(str)})


def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reclmis", required=True)
    ap.add_argument("--qata-dir", required=True, help="folder containing 'Test Set/Images' and 'Test Set/Ground-truths'")
    ap.add_argument("--mosmed-dir", required=True, help="folder containing frames/ and masks/ (original PNGs)")
    ap.add_argument("--lvit-dir", required=True)
    a = ap.parse_args()

    q = os.path.join(a.reclmis, "datasets", "Covid19", "Test_Folder")
    os.makedirs(q, exist_ok=True)
    link(os.path.join(a.qata_dir, "Test Set", "Images"), os.path.join(q, "img"))
    link(os.path.join(a.qata_dir, "Test Set", "Ground-truths"), os.path.join(q, "labelcol"))
    tq = norm(pd.read_excel(os.path.join(a.lvit_dir, "Test_text_for_Covid19.xlsx")))
    tq.to_excel(os.path.join(q, "Test_text.xlsx"), index=False)
    print("QaTa test reports:", len(tq))

    m = os.path.join(a.reclmis, "datasets", "MosMedplus", "Test_Folder")
    for sub in ("img", "labelcol"):
        os.makedirs(os.path.join(m, sub), exist_ok=True)
    tm = norm(pd.read_excel(os.path.join(a.lvit_dir, "Test_text_MosMedData+.xlsx")))
    miss = 0
    for n in tm.Image:
        f, g = os.path.join(a.mosmed_dir, "frames", n), os.path.join(a.mosmed_dir, "masks", n)
        if not (os.path.exists(f) and os.path.exists(g)):
            miss += 1
            continue
        link(f, os.path.join(m, "img", n)); link(g, os.path.join(m, "labelcol", n))
    tm.to_excel(os.path.join(m, "Test_text.xlsx"), index=False)
    print("MosMed test reports:", len(tm), "| missing image/mask pairs:", miss)


if __name__ == "__main__":
    main()
