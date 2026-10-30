# Laterality audit - VoxTell on ReXGroundingCT

**Verdict:** FOLLOWS_LATERALITY: swapped prompts rarely keep the original lesion

| quantity | value |
|---|---|
| scans / eligible findings / scored / hit | 200 / 177 / 174 / 126 |
| gate: lung control pass rate | 0.995 |
| gate: median GT fraction in thorax ROI | 1.000 |
| original-prompt Dice (mean, scored findings) | 0.300 |
| original-prompt hit rate | 0.724 |
| **Swap Failure Rate** | **0.008** (95% CI 0.000-0.026) |
| median stale overlap (swap / orig) | 0.000 |
| swap suppressed (prediction mostly vanishes) | 0.452 |
| swap mass on the swapped side (non-empty swaps) | 0.945 |
| Dice(swap, contralateral same-category lesion) | 0.172 |
| side-stripped prompt still hits | 0.753 |
| median cosine(orig, swap) text embedding | 0.927 |
| Spearman(cosine, stale overlap) | -0.093 (p=0.299) |

SFR by side: {"left": [0.0, 58], "right": [0.014705882352941176, 68]}

SFR by swap kind (side_word_only = only right/left or RUL-type words change): {"side_word_only": [0.0, 89], "anatomy_word": [0.02702702702702703, 37]}

SFR by lobe named: {"True": [0.008547008547008548, 117], "False": [0.0, 9]}

SFR by category: {"1b": [0.0, 4], "1c": [0.0, 1], "1e": [0.0, 1], "2a": [0.0, 26], "2b": [0.0, 26], "2c": [0.0, 21], "2d": [0.023255813953488372, 43], "2e": [0.0, 4]}

Excluded: {"text_mask_side_mismatch": 2, "outside_thorax": 1}

## Worst swap failures

| case | text | swap | stale | Dice orig |
|---|---|---|---|---|
| train_2464_a_1 | 4 mm indeterminate nodule in the medial subpleural region of the right middle lo | 4 mm indeterminate nodule in the medial subpleural region of the lingula | 0.926 | 0.554 |
| train_13148_a_2 | Two subpleural subcentimeter nodules measuring 4 mm at the basal level in the su | Two subpleural subcentimeter nodules measuring 4 mm at the basal level in the su | 0.407 | 0.573 |
| train_18403_a_2 | Right middle lobe opacification with air bronchograms | lingula opacification with air bronchograms | 0.162 | 0.672 |
| train_2532_a_2 | Patchy nodular ground-glass opacities in the right lung, predominantly subpleura | Patchy nodular ground-glass opacities in the left lung, predominantly subpleural | 0.161 | 0.463 |
| train_13481_a_2 | Areas of pneumonic consolidation with air bronchograms in the middle lobe | Areas of pneumonic consolidation with air bronchograms in the lingula | 0.145 | 0.623 |
| train_13398_a_1 | Parenchymal changes with scarring in the right middle lobe medial segment | Parenchymal changes with scarring in the lingula medial segment | 0.066 | 0.216 |
| train_19209_a_2 | Atelectasis in the inferior lingular segment of the left upper lobe and basal se | Atelectasis in the inferior right middle lobe and basal segments of the right lo | 0.020 | 0.552 |
| train_19771_a_2 | Subsegmental atelectasis in the medial segment of the right middle lobe | Subsegmental atelectasis in the medial segment of the lingula | 0.001 | 0.203 |
| train_18733_a_1 | Large consolidation with air bronchograms involving the posterobasal, mediobasal | Large consolidation with air bronchograms involving the posterobasal, mediobasal | 0.000 | 0.670 |
| train_13082_a_1 | Focal consolidative density at the paramediastinal level of the right middle lob | Focal consolidative density at the paramediastinal level of the lingula | 0.000 | 0.341 |
| train_13591_a_1 | Patchy ground glass densities in the inferior lingula of the left upper lobe | Patchy ground glass densities in the inferior right middle lobe | 0.000 | 0.488 |
| train_13591_a_1 | Atelectatic changes in the middle lobe of the right lung | Atelectatic changes in the lingula of the left lung | 0.000 | 0.240 |
| train_13177_a_2 | Atelectatic changes causing mild structural distortion in the left lung inferior | Atelectatic changes causing mild structural distortion in the right lung inferio | 0.000 | 0.540 |
| train_13301_a_2 | Subcentimeter calcific nodule in the upper lobe of the right lung | Subcentimeter calcific nodule in the upper lobe of the left lung | 0.000 | 0.451 |
| train_13583_d_2 | Progressive pleural effusion reaching a diameter of 3 cm between the leaves of t | Progressive pleural effusion reaching a diameter of 3 cm between the leaves of t | 0.000 | 0.658 |
