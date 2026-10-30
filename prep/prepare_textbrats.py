"""Make a BraTS 2020 copy match the file names in TextBraTS' json list.

Some public BraTS 2020 mirrors ship uncompressed .nii files, while TextBraTS' json expects .nii.gz, and
in one training case the segmentation file has a non-standard name. This script walks every path
string ending in .nii.gz in the json, and for each missing file looks in --src for the same file as
.nii (gzipped into place) or, for segmentations, any file in the case folder whose name contains
'seg'. It never overwrites existing files and reports anything it cannot resolve.

  python prepare_textbrats.py --json $EXT/TextBraTS/data/TextBraTSData/Test.json \
      --data-dir $EXT/TextBraTS/data/TextBraTSData --src $DATA/brats2020/MICCAI_BraTS2020_TrainingData
"""
import argparse, glob, gzip, json, os, shutil


def paths(o):
    if isinstance(o, str):
        if o.endswith(".nii.gz"):
            yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from paths(v)
    elif isinstance(o, list):
        for v in o:
            yield from paths(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, nargs="+")
    ap.add_argument("--data-dir", required=True, help="root the json paths are relative to")
    ap.add_argument("--src", required=True, help="BraTS 2020 training folder (case subfolders)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    want = sorted({p for j in a.json for p in paths(json.load(open(j)))})
    done = missing = 0
    for rel in want:
        dst = os.path.join(a.data_dir, rel)
        if os.path.exists(dst):
            continue
        case = os.path.basename(os.path.dirname(dst))
        name = os.path.basename(dst)
        cands = glob.glob(os.path.join(a.src, "**", name[:-3]), recursive=True) + \
                glob.glob(os.path.join(a.src, "**", name), recursive=True)
        if not cands and "seg" in name:
            cands = [p for p in glob.glob(os.path.join(a.src, "**", case, "*"), recursive=True)
                     if "seg" in os.path.basename(p).lower() and p.endswith((".nii", ".nii.gz"))]
        if len(cands) != 1:
            print(f"UNRESOLVED ({len(cands)} candidates): {rel}")
            missing += 1
            continue
        src = cands[0]
        print(f"{src} -> {dst}")
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if src.endswith(".gz"):
                shutil.copyfile(src, dst)
            else:
                with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        done += 1
    print(f"{len(want)} files listed, {done} created, {missing} unresolved")


if __name__ == "__main__":
    main()
