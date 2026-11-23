# DAPR-FRFN and PCGRC for Industrial Surface Defect Segmentation

This repository contains the implementation accompanying the manuscript **“Distribution-Aware Feature Refinement and Prediction-Guided Structural Correction for Industrial Surface Defect Segmentation.”** It is based on MMSegmentation 1.2.2 and includes the proposed model, controlled ablation variants, dataset definitions, fixed data splits, experiment configurations, and analysis scripts.

Model checkpoints are not distributed. All experiments can be trained from the released configurations, and the standard MMSegmentation evaluation entry point can be used with the checkpoint produced by each training run.

## Included experiments

- Main model on KSDD2, MTD, and NEU-SEG.
- Component ablations for FRFN, DAPR-FRFN, and PCGRC.
- DAPR internal ablations: NoStats, Mean-only, Stats-CA, and Stats-CA + PConv.
- PCGRC internal ablations: naive correction and scale-only correction.
- Configurations for the seven external baselines reported in the manuscript. External baseline implementations and checkpoints are not redistributed.

## Repository layout

- `projects/industrial_defect_seg/`: proposed models, ablation variants, datasets, and experiment configurations.
- `configs/`: MMSegmentation base configurations required by the experiment files.
- `tools/`: training, evaluation, and dataset-preparation entry points.
- `paper_analysis/`: efficiency, qualitative, routing, and failure-case analysis scripts.
- `data/*/splits/`: fixed split manifests. Raw datasets are not included.
- `results/expected_metrics.csv`: configuration-to-result mapping for the manuscript experiments.
- `docs/REPRODUCTION.md`: environment, data, training, and evaluation instructions.
- `docs/EXPERIMENTS.md`: experiment and configuration index.
- `NOTICE.md`: upstream attribution and third-party notices.

## Quick start

Create the environment and arrange the datasets as described in `docs/REPRODUCTION.md`. Run commands from the repository root.

```bash
python tools/train.py projects/industrial_defect_seg/configs/ksdd2/segformer_mit-b0_dapr_frfn_pcgrc_g001_ksdd2_20k.py
```

After training, evaluate any generated checkpoint with:

```bash
python tools/test.py CONFIG CHECKPOINT
```

Check the release tree, fixed split counts, and experiment-to-configuration mappings with:

```bash
python tools/verify_release.py
```

The fixed split sizes are:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| KSDD2 | 2,098 | 234 | 1,004 |
| MTD | 856 | 214 | 270 |
| NEU-SEG | 2,904 | 726 | 840 |

## Environment

The verified environment uses Python 3.10, PyTorch 2.1.0 with CUDA 12.1, MMCV 2.1.0, MMEngine 0.10.7, and MMSegmentation 1.2.2. See `docs/REPRODUCTION.md` for installation commands.

## Data and pretrained backbones

Raw KSDD2, MTD, and NEU-SEG data are not redistributed. Users must obtain them from their official sources and arrange them according to the documented directory layout. The fixed split manifests in this repository define the protocols used in the manuscript.

Some upstream baseline and backbone configurations refer to official OpenMMLab pretrained weights by URL. These initialization weights are third-party dependencies and are distinct from trained checkpoints of the proposed model.

## Checkpoint policy

Trained checkpoints are intentionally not included in the public repository. This applies to the main model and all ablation experiments. The repository instead provides the implementation, fixed configurations, data splits, seeds, training schedules, evaluation entry point, and reported metrics required to reproduce the experiments.

## Citation

If you use this repository, please cite the associated manuscript. Machine-readable citation metadata are provided in `CITATION.cff`.

## License and attribution

This repository is released under the Apache License 2.0 and retains the attribution required by the upstream projects. See `LICENSE`, `NOTICE.md`, and `CITATION.cff`.
