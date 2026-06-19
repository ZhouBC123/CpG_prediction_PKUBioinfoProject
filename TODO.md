# TODO

## 0. Project Setup

- [x] Configure git ignore rules for generated data, model checkpoints, notebooks, and logs.
- [x] Add conda environment specification.
- [x] Create and validate the conda environment on the current machine.
- [x] Document project scope, model families, and raw data location.

## 1. Raw Data Download

- [x] Select an open data source for CpG island annotations and reference sequence.
- [x] Add a reproducible download script.
- [x] Download UCSC hg38 CpG island annotation, schema, reference genome, chromosome sizes, and checksums.
- [x] Verify downloaded file integrity and record local hashes.

## 2. Preprocessing

- [x] Parse UCSC `cpgIslandExt` table.
- [x] Build binary-classification windows.
- [x] Build base-level segmentation labels.
- [x] Split train/validation/test by chromosome to reduce leakage.

## 3. Models

- [x] Implement Markov chain baseline for binary classification.
- [x] Implement HMM baseline for base-level segmentation.
- [x] Implement fully connected neural network.
- [x] Implement CNN.
- [x] Add metrics and training scripts/notebooks.

## 4. Report

- [ ] Describe input/output data and acquisition method.
- [ ] Summarize implementation and running instructions.
- [x] Report experiments, tuning, and results.
