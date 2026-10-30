"""Data and config for retraining Ariadne's Thread (LanGuideMedSeg-MICCAI2023) on QaTa-COV19 and MosMedData+.

The repo validates on the LAST 20% of train.csv, so train.csv lists the LViT train reports followed by
the LViT validation reports (QaTa: the cut falls exactly on the LViT split, 5716/1429). Test sets and
reports are the same as for RecLMIS. MosMedData+ PNGs are stored as 3-channel; the repo expects
single-channel images and 0/255 masks like QaTa, so grayscale copies are written (the script checks that
the RGB channels are identical, i.e. nothing is lost). Writes config/training_mosmed.yaml.

  python prepare_ariadne.py --repo $EXT/LanGuideMedSeg-MICCAI2023 --qata-dir $DATA/qata/QaTa-COV19-v2 \
      --mosmed-dir $DATA/mosmed --mosmed-1ch $DATA/mosmed_1ch --lvit-dir $DATA/lvit_texts
"""
import argparse, os
import numpy as np, pandas as pd
from PIL import Image


def norm(df, prefix_mask=False):
    ic = "Image" if "Image" in df else df.columns[0]
    tc = "Description" if "Description" in df else [c for c in df.columns if c != ic][0]
    names = df[ic].astype(str).str.strip()
    if prefix_mask:
        names = names.map(lambda s: s if s.startswith("mask_") else "mask_" + s)
    return pd.DataFrame({"Image": names, "Description": df[tc].astype(str)})


def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def check(df, root, strip_prefix):
    bad = [n for n in df.Image if not (os.path.exists(os.path.join(root, "Images", n[5:] if strip_prefix else n))
                                       and os.path.exists(os.path.join(root, "GTs", n)))]
    return len(bad), bad[:3]


def qata(a):
    R = os.path.join(a.repo, "data", "QaTa-COV19-v2")
    for split, src in (("Train", "Train Set"), ("Test", "Test Set")):
        os.makedirs(os.path.join(R, split), exist_ok=True)
        link(os.path.join(a.qata_dir, src, "Images"), os.path.join(R, split, "Images"))
        link(os.path.join(a.qata_dir, src, "Ground-truths"), os.path.join(R, split, "GTs"))
    os.makedirs(os.path.join(R, "prompt"), exist_ok=True)
    L = a.lvit_dir
    tv = norm(pd.read_excel(f"{L}/Train_Val_text_for_Covid19.xlsx"), True).drop_duplicates("Image").set_index("Image")
    mk = lambda x: (lambda s: s if s.startswith("mask_") else "mask_" + s)(str(x).strip())
    tr = [mk(x) for x in pd.read_excel(f"{L}/Train_ID_for_Covid19.xlsx").iloc[:, 0]]
    va = [mk(x) for x in pd.read_excel(f"{L}/Val_ID_for_Covid19.xlsx").iloc[:, 0]]
    order = [n for n in tr + va if n in tv.index]
    train = pd.DataFrame({"Image": order, "Description": [tv.loc[n, "Description"] for n in order]})
    test = norm(pd.read_excel(f"{L}/Test_text_for_Covid19.xlsx"), True)
    train.to_csv(os.path.join(R, "prompt", "train.csv"), index=False)
    test.to_csv(os.path.join(R, "prompt", "test.csv"), index=False)
    print(f"QaTa: train {len(tr)} + val {len(va)} (missing text {len(tr) + len(va) - len(order)}), test {len(test)}; "
          f"80% cut = {int(0.8 * len(train))}; missing files train {check(train, os.path.join(R, 'Train'), True)}, "
          f"test {check(test, os.path.join(R, 'Test'), True)}")


def mosmed(a):
    names = sorted(os.listdir(os.path.join(a.mosmed_dir, "frames")))
    for sub in ("frames", "masks"):
        os.makedirs(os.path.join(a.mosmed_1ch, sub), exist_ok=True)
    neq = 0
    for n in names:
        im = np.array(Image.open(os.path.join(a.mosmed_dir, "frames", n)).convert("RGB"))
        mk = np.array(Image.open(os.path.join(a.mosmed_dir, "masks", n)).convert("RGB"))
        neq += int(not (np.array_equal(im[..., 0], im[..., 1]) and np.array_equal(im[..., 0], im[..., 2])))
        Image.fromarray(im[..., 0]).save(os.path.join(a.mosmed_1ch, "frames", n))
        Image.fromarray(((mk.max(-1) > 127) * 255).astype(np.uint8)).save(os.path.join(a.mosmed_1ch, "masks", n))
    print(f"MosMed: converted {len(names)} slices; images with unequal RGB channels: {neq} (should be 0)")
    R = os.path.join(a.repo, "data", "MosMedDataPlus")
    for split in ("Train", "Test"):
        os.makedirs(os.path.join(R, split), exist_ok=True)
        link(os.path.join(a.mosmed_1ch, "frames"), os.path.join(R, split, "Images"))
        link(os.path.join(a.mosmed_1ch, "masks"), os.path.join(R, split, "GTs"))
    os.makedirs(os.path.join(R, "prompt"), exist_ok=True)
    L = a.lvit_dir
    tr = norm(pd.read_excel(f"{L}/Train_text_MosMedData.xlsx"))
    va = norm(pd.read_excel(f"{L}/Val_text_MosMedData.xlsx"))
    te = norm(pd.read_excel(f"{L}/Test_text_MosMedData+.xlsx"))
    train = pd.concat([tr, va], ignore_index=True)
    train.to_csv(os.path.join(R, "prompt", "train.csv"), index=False)
    te.to_csv(os.path.join(R, "prompt", "test.csv"), index=False)
    print(f"MosMed: train {len(tr)} + val {len(va)}, test {len(te)}, train/test overlap "
          f"{len(set(train.Image) & set(te.Image))}; missing files {check(train, os.path.join(R, 'Train'), False)}, "
          f"{check(te, os.path.join(R, 'Test'), False)}")
    cfg_in = os.path.join(a.repo, "config", "training.yaml")
    s = open(cfg_in).read()
    for old, new in (("QaTa-COV19-v2/prompt/train.csv", "MosMedDataPlus/prompt/train.csv"),
                     ("QaTa-COV19-v2/prompt/test.csv", "MosMedDataPlus/prompt/test.csv"),
                     ("QaTa-COV19-v2/Train", "MosMedDataPlus/Train"), ("QaTa-COV19-v2/Test", "MosMedDataPlus/Test"),
                     ("model_save_filename: medseg", "model_save_filename: medseg_mosmed")):
        assert old in s, f"{old!r} not found in {cfg_in}"
        s = s.replace(old, new)
    open(os.path.join(a.repo, "config", "training_mosmed.yaml"), "w").write(s)
    print("wrote config/training_mosmed.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--qata-dir", required=True)
    ap.add_argument("--mosmed-dir", required=True)
    ap.add_argument("--mosmed-1ch", required=True, help="output folder for single-channel MosMed copies")
    ap.add_argument("--lvit-dir", required=True)
    ap.add_argument("--only", choices=["qata", "mosmed"])
    a = ap.parse_args()
    if a.only != "mosmed":
        qata(a)
    if a.only != "qata":
        mosmed(a)


if __name__ == "__main__":
    main()
