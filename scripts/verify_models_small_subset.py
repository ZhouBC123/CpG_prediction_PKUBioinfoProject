#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cpg_prediction.data import load_npz_arrays, take_small_subset  # noqa: E402
from cpg_prediction.neural import (  # noqa: E402
    SequenceCNNClassifier,
    SequenceCNNSegmenter,
    SequenceMLPClassifier,
    SequenceMLPSegmenter,
)
from cpg_prediction.probabilistic import MarkovBinaryClassifier, SupervisedHMMCpGSegmenter  # noqa: E402
from cpg_prediction.training import train_torch_smoke  # noqa: E402


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--binary-samples", type=int, default=128)
    parser.add_argument("--segmentation-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    x_binary, y_binary = load_npz_arrays(args.processed_dir, "binary", "train")
    x_seg, y_seg = load_npz_arrays(args.processed_dir, "segmentation", "train")
    x_binary, y_binary = take_small_subset(
        x_binary,
        y_binary,
        max_samples=args.binary_samples,
        seed=args.seed,
        balanced_binary=True,
    )
    x_seg, y_seg = take_small_subset(
        x_seg,
        y_seg,
        max_samples=args.segmentation_samples,
        seed=args.seed,
        balanced_binary=False,
    )

    summary: dict[str, object] = {
        "binary_subset": {"X": list(x_binary.shape), "y": list(y_binary.shape), "positives": int(y_binary.sum())},
        "segmentation_subset": {
            "X": list(x_seg.shape),
            "y": list(y_seg.shape),
            "positive_bases": int(y_seg.sum()),
        },
        "models": {},
    }

    markov = MarkovBinaryClassifier().fit(x_binary, y_binary)
    markov_probs = markov.predict_proba(x_binary[:8])
    markov_pred = markov.predict(x_binary[:8])
    assert markov_probs.shape == (8, 2)
    assert np.allclose(markov_probs.sum(axis=1), 1.0)
    assert set(markov_pred.tolist()) <= {0, 1}
    summary["models"]["markov_binary"] = {
        "proba_shape": list(markov_probs.shape),
        "prediction_sample": markov_pred.tolist(),
    }

    hmm = SupervisedHMMCpGSegmenter().fit(x_seg, y_seg)
    hmm_pred = hmm.predict(x_seg[:8])
    assert hmm_pred.shape == y_seg[:8].shape
    assert set(np.unique(hmm_pred).tolist()) <= {0, 1}
    summary["models"]["hmm_segmentation"] = {
        "prediction_shape": list(hmm_pred.shape),
        "positive_bases_predicted": int(hmm_pred.sum()),
    }

    binary_loader = make_loader(x_binary, y_binary, args.batch_size)
    seg_loader = make_loader(x_seg, y_seg, args.batch_size)
    loss_fn = nn.BCEWithLogitsLoss()

    torch_models = {
        "mlp_binary": (SequenceMLPClassifier(sequence_length=x_binary.shape[1]), binary_loader),
        "cnn_binary": (SequenceCNNClassifier(), binary_loader),
        "mlp_segmentation": (SequenceMLPSegmenter(sequence_length=x_seg.shape[1]), seg_loader),
        "cnn_segmentation": (SequenceCNNSegmenter(), seg_loader),
    }

    for name, (model, loader) in torch_models.items():
        result = train_torch_smoke(model, loader, loss_fn=loss_fn, steps=args.steps, device="cpu")
        summary["models"][name] = {
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "output_shape": list(result.output_shape),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
