# Do Text-Guided Medical Image Segmenters Read the Text? A Counterfactual Audit

Code and result files for the ISBI 2027 submission. The audit re-evaluates text-guided segmentation models on their
own test splits with the report replaced or edited, and measures how much the segmentation changes:

| condition | what the model receives |
|---|---|
| matched | the case's own report |
| constant | one case-independent input for every case (mean text feature, or the most frequent report) |
| shuffled | another case's report with different wording (3 seeds; drawn independently per case for RecLMIS and Ariadne's Thread, a derangement for TextBraTS) |
| slot edits | one attribute changed: side (left/right), extent (all/upper/middle/lower), lesion count, uni-/bilateral |
| paraphrase | meaning-preserving rewording, controlled so that no report is newly truncated by the tokenizer |

The effect Delta = Dice(matched) - Dice(condition) is reported with 95% paired bootstrap CIs (5,000 resamples),
clustered by patient (BraTS; QaTa-COV19 subject ids from the file names) or CT volume (MosMedData+: 273 test
slices from only 9 volumes). The VoxTell failure rate uses 2,000 scan-level resamples. A model *uses text* if
Delta >= 1 Dice point with CI lower bound > 0, and is *equivalent* (negligible effect) if the whole CI lies within
+-0.5 points.

Audited settings: TextBraTS (BraTS 2020 MRI, released checkpoint), RecLMIS (QaTa-COV19 X-ray and MosMedData+ CT,
released checkpoints), Ariadne's Thread (both datasets, retrained with the official code), and a prompt-level
laterality audit of VoxTell on ReXGroundingCT CT.

## Repository layout

```
audit/               counterfactual runners (imported from inside each model's own repository)
  shuffle_test.py      TextBraTS: matched / mean-feature / shuffled reports
  tb_agree.py          TextBraTS: prediction-level agreement (identical masks, voxel flips, rerun determinism)
  reclmis_shuffle.py   RecLMIS: matched / constant / shuffled / left-right swap / zero-mask label check
  reclmis_slots.py     RecLMIS: single-slot edits (side, count, extent, uni-/bilateral)
  reclmis_nomask.py    RecLMIS: is the ground-truth mask passed to the test-time forward used? (+ timing)
  ariadne_cf.py        Ariadne's Thread: all conditions and slot edits
  para.py              paraphrase rules (meaning-preserving rewrites of the LViT report template)
  para_reclmis.py, para_ariadne.py   paraphrase runs
  para_lenmatch.py     how the truncation-controlled opening rewrite was chosen (RecLMIS truncates at 18 CLIP tokens)
  para_untrunc.py      RecLMIS paraphrase restricted to reports never truncated
  para_rules.py        one paraphrase rule at a time
  lat_check.py         do report side words agree with the image side of the mask?
voxtell_laterality/  VoxTell laterality audit (own README)
prep/                data layout and compatibility patches for the third-party repositories
analysis/            statistics behind the paper's text (side-word statistics, token limits, VoxTell
                     bookkeeping, TextBraTS equivalence CIs) and make_tables.py, which recomputes Table 1
paper/make_fig.py    Fig. 1
patches/             exact diffs we applied to Ariadne's Thread, and the upstream commits used
requirements/        pip freeze of the two environments used
results/             the result files behind every number in the paper (see below)
```

## Reproduce the paper's numbers from the included results

```bash
python analysis/make_tables.py --results results     # Table 1 Dice/Delta (image-level CIs), paraphrase rules, zero-mask check
python analysis/cluster_ci.py --results results      # patient/volume-clustered CIs: the intervals reported in the paper
python analysis/voxtell_summary.py --work results/lat_full
python analysis/tb_delta_ci.py --agree results/textbrats_agree/agree.json
cd paper && python make_fig.py                       # fig_slots.pdf
```

## Rerun everything

### 1. Environments

Two environments were used (exact versions in `requirements/`; GPU: one NVIDIA RTX A4500):

* **main** (Python 3.12, torch 2.14.1+cu126): TextBraTS, RecLMIS, VoxTell and all analysis scripts.
  Needs `monai`, `opencv-python-headless`, `openpyxl`, `ml_collections`, `ftfy`, `regex` and OpenAI CLIP
  (`pip install git+https://github.com/openai/CLIP.git`).
* **ariadne** (Python 3.10, torch 2.0.1+cu118, pytorch-lightning 1.9.5, transformers 4.30.2, monai 1.2.0,
  `setuptools<70`): Ariadne's Thread, which predates Lightning 2.
* TotalSegmentator needs its own environment (see `voxtell_laterality/README.md`).

### 2. Third-party code, checkpoints and data

Clone the model repositories (commits in `patches/upstream_commits.txt`) under a folder `$EXT`:
[RecLMIS](https://github.com/ShawnHuang497/RecLMIS),
[LanGuideMedSeg-MICCAI2023](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023) (Ariadne's Thread),
the official TextBraTS code (Hugging Face model card; dataset `Jupitern52/TextBraTS`), and install
[VoxTell](https://pypi.org/project/voxtell/) 0.1.2 (weights `mrokuss/VoxTell`, snapshot `4e3afbac`; see `patches/upstream_commits.txt`).

* TextBraTS: authors' checkpoint at `runs/TextBraTS/model.pt`; BraTS 2020 training data merged with the authors'
  TextBraTS reports and report features into `data/TextBraTSData`. If your BraTS copy has `.nii` files or a
  non-standard segmentation name, run `prep/prepare_textbrats.py`.
* RecLMIS: the released QaTa-COV19 and MosMedData+ checkpoints, each in its own folder
  (`ckpt/qata/`, `ckpt/mosmed/`; both files have the same name), and CLIP `ViT-B-32.pt` in `nets/`.
* Data: QaTa-COV19-v2, MosMedData+ (frames/ and masks/), and the LViT text annotations
  (`Train_Val_text_for_Covid19.xlsx`, `Train_ID_for_Covid19.xlsx`, `Val_ID_for_Covid19.xlsx`,
  `Test_text_for_Covid19.xlsx`, `Train_text_MosMedData.xlsx`, `Val_text_MosMedData.xlsx`,
  `Test_text_MosMedData+.xlsx`). ReXGroundingCT and CT-RATE for VoxTell (gated on Hugging Face).

```bash
export AUDIT=$PWD/audit PREP=$PWD/prep OUT=/path/to/outputs DATA=/path/to/data EXT=/path/to/repos
python $PREP/prepare_reclmis.py --reclmis $EXT/RecLMIS --qata-dir $DATA/qata/QaTa-COV19-v2 \
    --mosmed-dir $DATA/mosmed --lvit-dir $DATA/lvit_texts
python $PREP/patch_ariadne.py --repo $EXT/LanGuideMedSeg-MICCAI2023          # see patches/ for the exact diff
python $PREP/prepare_ariadne.py --repo $EXT/LanGuideMedSeg-MICCAI2023 --qata-dir $DATA/qata/QaTa-COV19-v2 \
    --mosmed-dir $DATA/mosmed --mosmed-1ch $DATA/mosmed_1ch --lvit-dir $DATA/lvit_texts
```

`prepare_ariadne.py` writes the training files so that the repository's built-in validation split (last 20% of
`train.csv`) equals the LViT split on QaTa-COV19, converts the 3-channel MosMedData+ PNGs to single-channel copies
(it checks that no information is lost) and writes `config/training_mosmed.yaml`.

### 3. Runs (each from inside the model's repository)

```bash
# TextBraTS
cd $EXT/TextBraTS
A="--data_dir ./data/TextBraTSData --json_list Test.json --ckpt ./runs/TextBraTS/model.pt"
python $AUDIT/shuffle_test.py $A --out $OUT/textbrats_kt
python $AUDIT/tb_agree.py     $A --out $OUT/textbrats_agree

# RecLMIS (QaTa: Config_covid19 + ckpt/qata; MosMedData+: Config_MosMedPlus + ckpt/mosmed)
cd $EXT/RecLMIS
Q="--cfg Config_covid19 --ckpt ckpt/qata/best_model-RecLMIS.pth.tar"
M="--cfg Config_MosMedPlus --ckpt ckpt/mosmed/best_model-RecLMIS.pth.tar"
python $AUDIT/reclmis_shuffle.py $Q --out $OUT/reclmis_qata;   python $AUDIT/reclmis_shuffle.py $M --out $OUT/reclmis_mosmed
python $AUDIT/reclmis_slots.py   $Q --out $OUT/slots_qata;     python $AUDIT/reclmis_slots.py   $M --out $OUT/slots_mosmed
python $AUDIT/para_lenmatch.py   $Q --outbase $OUT/para_lenmatch          # QaTa paraphrase (all_lenmatched)
python $AUDIT/para_reclmis.py    $M --out $OUT/para2_reclmis_mosmed
python $AUDIT/para_untrunc.py    $Q --out $OUT/para_untrunc_qata
python $AUDIT/para_rules.py reclmis $Q --outbase $OUT/pararules_reclmis_qata
python $AUDIT/reclmis_nomask.py  $Q --out $OUT/nomask_qata;    python $AUDIT/reclmis_nomask.py  $M --out $OUT/nomask_mosmed

# Ariadne's Thread (ariadne env): train, then audit
cd $EXT/LanGuideMedSeg-MICCAI2023
python train.py --config ./config/training.yaml                 # -> save_model/medseg.ckpt
python train.py --config ./config/training_mosmed.yaml          # -> save_model/medseg_mosmed.ckpt
python evaluate_ckpt.py --config ./config/training_mosmed.yaml --ckpt ./save_model/medseg_mosmed.ckpt
python $AUDIT/ariadne_cf.py --out $OUT/ariadne_qata
python $AUDIT/ariadne_cf.py --config ./config/training_mosmed.yaml --ckpt ./save_model/medseg_mosmed.ckpt --out $OUT/ariadne_mosmed
python $AUDIT/para_ariadne.py --out $OUT/para2_ariadne_qata
python $AUDIT/para_ariadne.py --config ./config/training_mosmed.yaml --ckpt ./save_model/medseg_mosmed.ckpt --out $OUT/para2_ariadne_mosmed
python $AUDIT/para_rules.py ariadne --outbase $OUT/pararules_ariadne_qata

# Annotation and tokenizer statistics
python analysis/side_stats.py --lvit-dir $DATA/lvit_texts --qata-dir $DATA/qata/QaTa-COV19-v2 --mosmed-masks $DATA/mosmed_1ch/masks
python analysis/tok_check.py --xlsx QaTa=$EXT/RecLMIS/datasets/Covid19/Test_Folder/Test_text.xlsx \
                             --xlsx MosMed=$EXT/RecLMIS/datasets/MosMedplus/Test_Folder/Test_text.xlsx   # run in both envs
```

VoxTell: see `voxtell_laterality/README.md`.

## Notes on the third-party code

* **RecLMIS** passes the ground-truth mask to the forward call at test time; `reclmis_nomask.py` shows the
  outputs are bit-identical with a zero mask. Its test mode raises `UnboundLocalError` after computing the
  prediction, so `forward_logits` in `reclmis_shuffle.py` reads the output through a forward hook. Its report
  padding uses a pandas chained assignment that is a silent no-op under pandas >= 3 (copy-on-write); our
  runners re-apply the identical padding rule and check the result. Reports are truncated to 18 CLIP tokens,
  which cuts 24.6% of the QaTa-COV19 test reports.
* **Ariadne's Thread** needed one compatibility change to train under PyTorch Lightning 1.9 (hyper-parameter
  access in `engine/wrapper.py`); data-loader workers were reduced to 8. Model and training logic are unchanged
  (`patches/ariadne/`).
* All runners stop unless the checkpoint matches the model exactly (0 missing / 0 unexpected keys). During the
  study the RecLMIS and TextBraTS runs logged 0 / 0 and Ariadne's Thread was loaded with `strict=True`; the
  released runners enforce this instead of only reporting it.

## Result files

`results/` holds the outputs behind the paper: `summary.json` per run, per-image or per-case Dice where
available, `edits.json` (every edited report), `textbrats_agree/agree.json` (prediction-level agreement) and
`lat_full/` (VoxTell per-finding table), and `annotation_stats/` (saved outputs of `analysis/side_stats.py` and
`analysis/tok_check.py`; the LViT annotation files themselves are not redistributed). In the VoxTell results a
*suppressed* swap means the swapped prediction has < 10% of the original prediction's volume (not necessarily
empty), and left/right shares are computed over predicted voxels inside the thoracic ROI. Two caveats:

* `pararules_*/rule0` used the first version of the opening rewrite ("Infection of both lungs"), which pushes
  98.7% of affected reports past RecLMIS' 18-token limit. `para.py` now uses the truncation-controlled
  "Bilateral lung infection" (chosen by `para_lenmatch.py`); rerunning `para_rules.py` reproduces rules 1-5
  but gives the controlled value (0.13 points for RecLMIS) for rule 0.
* `results/superseded/` holds the first, truncation-confounded paraphrase runs (RecLMIS 30.2 points on QaTa),
  reported in the paper only as the pitfall that motivated the truncation control.

Verdict labels stored in the summaries were renamed to the neutral names used in the code; thresholds are
unchanged. The paper's classification (two-sided equivalence) is recomputed by `analysis/make_tables.py`.
The VoxTell scan sources and training-list overlap (200 scans: 50 base-validation + 150 challenge-validation,
0 in the ReXGroundingCT training list) come from `scans.jsonl`, which `voxtell_laterality/prepare.py` writes
into the work folder (not included here because it holds local file paths).

## License

MIT for the code in this repository (see `LICENSE`). The audited models, datasets and checkpoints keep their
own licenses; `patches/` contains diffs against third-party code distributed under its original license.
