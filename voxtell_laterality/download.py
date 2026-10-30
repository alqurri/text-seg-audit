"""Fetch only what the laterality audit needs: the ReXGroundingCT repo (~3.6 GB) and the matching CT-RATE volumes.

Both Hugging Face datasets are gated: accept the terms on their pages first and export HF_TOKEN.

  python download.py --rex-dir $D/rexgroundingct --ctrate-dir $D/ctrate \
      --source dataset.json:val --source MICCAI_challenge_dataset.json:val [--limit 5] [--dry-run]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ctrate_relpaths, index_files, load_entries  # noqa: E402

REX_REPO = "rajpurkarlab/ReXGroundingCT"
CTRATE_REPO = "ibrahimhamamci/CT-RATE"
CTRATE_META = ["dataset/metadata/train_metadata.csv", "dataset/metadata/validation_metadata.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rex-dir", required=True)
    ap.add_argument("--ctrate-dir", required=True)
    ap.add_argument("--source", action="append", default=[], help="json_file:split (repeatable)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases per source (smoke test)")
    ap.add_argument("--skip-rex", action="store_true", help="ReXGroundingCT already downloaded")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.source:
        a.source = ["dataset.json:val", "MICCAI_challenge_dataset.json:val"]

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError

    if not a.skip_rex and not a.dry_run:
        print(f"[rex] snapshot {REX_REPO} -> {a.rex_dir}")
        snapshot_download(REX_REPO, repo_type="dataset", local_dir=a.rex_dir)

    names = []
    for spec in a.source:
        jf, split = spec.split(":")
        entries = load_entries(os.path.join(a.rex_dir, jf), split)
        if a.limit:
            entries = entries[: a.limit]
        names += [e["name"] for e in entries]
    names = sorted(set(names))
    have = index_files(a.ctrate_dir) if os.path.isdir(a.ctrate_dir) else {}
    todo = [n for n in names if n.replace(".nii.gz", "") not in have]
    print(f"[ctrate] {len(names)} volumes needed, {len(names) - len(todo)} present, {len(todo)} to fetch")
    if a.dry_run:
        for n in todo[:10]:
            print("   ", n, "->", ctrate_relpaths(n))
        return

    for meta in CTRATE_META:
        try:
            hf_hub_download(CTRATE_REPO, meta, repo_type="dataset", local_dir=a.ctrate_dir)
        except Exception as e:  # metadata is only used for a sanity check
            print(f"[ctrate] metadata {meta} not fetched: {e}")

    failed = []
    for i, n in enumerate(todo, 1):
        for rel in ctrate_relpaths(n):
            try:
                hf_hub_download(CTRATE_REPO, rel, repo_type="dataset", local_dir=a.ctrate_dir)
                print(f"[{i}/{len(todo)}] {rel}")
                break
            except EntryNotFoundError:
                continue
        else:
            failed.append(n)
            print(f"[{i}/{len(todo)}] NOT FOUND {n}")
    if failed:
        print(f"{len(failed)} volumes not found; first: {failed[:5]}")


if __name__ == "__main__":
    main()
