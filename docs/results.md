# Experiment Results

This file is generated from `results/metrics.json` after running `scripts/run_experiments.py`.

## Setup

- Device: `cuda`
- Seed: `20260619`
- Processed data: `processed_data`
- Neural tuning limits: binary `12000`, segmentation `8000`
- Full neural training uses the complete train split after selecting the best tuning config per model family.

## Best Validation Choices

| Task | Model | Phase | Train samples | Selected config | Validation F1 | Test F1 | Other test metrics |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| binary | Markov | full | 46258 | `alpha=0.1` | 0.9598 | 0.9616 | test AUROC=0.9884 |
| segmentation | HMM | full | 46258 | `alpha=0.1` | 0.8103 | 0.8181 | test IoU=0.6922 |
| binary | MLP | full | 46258 | `mlp_binary_h256_lr5e-4; selected from tuning mlp_binary_h256_lr5e-4` | 0.9665 | 0.9607 | test AUROC=0.9932 |
| binary | CNN | full | 46258 | `cnn_binary_c64_lr5e-4; selected from tuning cnn_binary_c64_lr5e-4` | 0.9871 | 0.9908 | test AUROC=0.9994 |
| segmentation | MLP | full | 46258 | `mlp_segmentation_h256_lr5e-4; selected from tuning mlp_segmentation_h256_lr5e-4` | 0.8643 | 0.8733 | test IoU=0.7751 |
| segmentation | CNN | full | 46258 | `cnn_segmentation_c64_lr5e-4; selected from tuning cnn_segmentation_c64_lr5e-4` | 0.7849 | 0.7868 | test IoU=0.6486 |

## Notes

- Markov and HMM baselines are fit with Laplace smoothing grid search over alpha values.
- Neural candidates are first checked on deterministic smaller training subsets, then the best MLP/CNN candidate per task is retrained on the full training split.
- Checkpoints are written under `models/` and intentionally ignored by git.
