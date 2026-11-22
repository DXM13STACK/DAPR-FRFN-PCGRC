# Experiment index

## Main model

| Dataset | Configuration |
|---|---|
| KSDD2 | `projects/industrial_defect_seg/configs/ksdd2/segformer_mit-b0_dapr_frfn_pcgrc_g001_ksdd2_20k.py` |
| MTD | `projects/industrial_defect_seg/configs/mtd/segformer_mit-b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4.py` |
| NEU-SEG | `projects/industrial_defect_seg/configs/neu/segformer_mit-b0_dapr_frfn_pcgrc_g001_neu_seg_20k.py` |

## Component ablations

FRFN, DAPR-FRFN, and the complete DAPR-FRFN + PCGRC model are provided for all three datasets. The configuration-to-result mapping is recorded in `results/expected_metrics.csv`.

## DAPR internal ablations

The KSDD2 and MTD configuration directories include:

- NoStats routing
- Mean-only routing
- Stats-CA
- Stats-CA + PConv

## PCGRC internal ablations

The KSDD2 and MTD configuration directories include:

- Naive correction
- Scale-only correction
- Full prediction-conditioned gated residual correction

The KSDD2 naive-correction configuration uses the same explicit correction implementation as the MTD experiment. Its final test result is included in `results/expected_metrics.csv`.

## External comparisons

The dataset configuration directories include the seven external comparison models reported in the manuscript. These files document the data protocol and training settings. External source code and trained weights are not repackaged in this repository.

## Output policy

Training outputs are written to `work_dirs/` and are ignored by Git. The public repository contains no `.pth` files, checkpoint paths, binary hashes, or promises of downloadable trained weights.
