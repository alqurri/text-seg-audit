# Laterality audit: does VoxTell follow side words in free-text prompts?

For every ReXGroundingCT finding that names one side (e.g. *"nodule in the right upper lobe"*), VoxTell is
prompted with the original text and with the side swapped (*"left upper lobe"*; right middle lobe <-> lingula,
RUL <-> LUL, ...). We then ask whether the swapped prompt still segments the **original** lesion.

**Metric.** A finding is a *hit* if the original prompt covers >= 10% of its ground-truth lesion. A swap
*fails* if the swapped prompt still covers >= 50% of the voxels the original prompt hit. Swap Failure Rate
(SFR) = failures / hits, with a bootstrap CI over scans. This measures retention of the original lesion,
not overall correctness; `analyze.py` also reports how often the swapped prediction vanishes, how much of it
lies on the named side, and its overlap with same-category contralateral lesions (identified from the
scan's ground-truth masks and assigned to a side by TotalSegmentator lung lobes).

**Controls.** Side assignment never relies on header conventions: left/right comes from TotalSegmentator
lobes (each voxel goes to the nearer lung, +-25 mm ROI). A "left lung" / "right lung" prompt control must
land on the correct side in >= 95% of scans, otherwise the run aborts (input orientation bug).

## Data

Scans with released masks: ReXGroundingCT `dataset.json` validation and `MICCAI_challenge_dataset.json`
validation; the paper's run used 200 unique scans (50 + 150), none of which appear in the ReXGroundingCT
training list. Both datasets are gated on Hugging Face: accept the terms for
[ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) and
[CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), then `hf auth login`
(never put the token in a script or command line). Only the needed CT-RATE volumes are downloaded.

## Environments

Two environments avoid nnU-Net version clashes (see `../requirements/` for the exact versions used).

```bash
conda create -y -n voxtell python=3.12 && conda activate voxtell
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install voxtell scipy pytest

conda create -y -n totalseg python=3.11 && conda activate totalseg
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install TotalSegmentator
```

## Run

```bash
export DATA=/path/to/data  W=$DATA/voxtell_work  HF_HOME=$DATA/hf_cache
python -m pytest -q tests                                                    # synthetic data, mock VoxTell
python download.py --rex-dir $DATA/rexgroundingct --ctrate-dir $DATA/ctrate   # voxtell env
python prepare.py  --rex-dir $DATA/rexgroundingct --ctrate-dir $DATA/ctrate --work $W
python lungs.py --work $W --fast                                              # totalseg env
python infer_voxtell.py --work $W                                             # voxtell env
python analyze.py --work $W && cat $W/results/report.md
python ../analysis/voxtell_summary.py --work $W      # funnel, exact CIs, outcome breakdown (paper numbers)
```

Or run all steps with `VX=<voxtell python> TS=<totalseg python> DATA=... W=... bash run_full.sh`.
Smoke test first with `prepare.py --limit 3` into a separate `--work`. Every step resumes (finished scans
are skipped). Runtime (rough): TotalSegmentator `--fast` 0.5-1 min per scan, VoxTell a few minutes per chest CT.

Outputs: `results/report.md`, `summary.json`, `per_finding.csv` (every finding with its swap, overlaps and
text-embedding cosine) and `per_scan.csv` (controls). The paper's outputs are in `../results/lat_full/`.
