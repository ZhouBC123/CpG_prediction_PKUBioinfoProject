# Models

Implemented model families:

- `MarkovBinaryClassifier`: class-conditional first-order Markov model for binary CpG island classification.
- `SupervisedHMMCpGSegmenter`: two-state supervised HMM for base-level CpG island segmentation.
- `SequenceMLPClassifier`: fully connected binary classifier over one-hot encoded fixed windows.
- `SequenceMLPSegmenter`: fully connected base-level segmenter over one-hot encoded fixed windows.
- `SequenceCNNClassifier`: 1D CNN binary classifier.
- `SequenceCNNSegmenter`: fully convolutional 1D CNN segmenter.

Run a small correctness verification:

```bash
/root/miniconda3/envs/cpg-prediction/bin/python scripts/verify_models_small_subset.py
```

The verification uses a small training subset, checks probability/prediction shapes for the probabilistic models, and runs a few optimization steps for each neural model to verify forward pass, loss computation, backward pass, and parameter update.
