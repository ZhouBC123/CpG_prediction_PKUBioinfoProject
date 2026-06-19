from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Task = Literal["binary", "segmentation"]
Split = Literal["train", "val", "test"]


class NpzSequenceDataset(Dataset):
    """PyTorch dataset for processed CpG npz files."""

    def __init__(self, path: str | Path, task: Task):
        data = np.load(path)
        self.x = data["X"].astype(np.int64, copy=False)
        self.y = data["y"].astype(np.float32, copy=False)
        self.task = task

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[index])
        y = torch.from_numpy(np.asarray(self.y[index]))
        return x, y


def load_npz_arrays(processed_dir: str | Path, task: Task, split: Split) -> tuple[np.ndarray, np.ndarray]:
    path = Path(processed_dir) / task / f"{split}.npz"
    data = np.load(path)
    return data["X"].astype(np.uint8, copy=False), data["y"].astype(np.uint8, copy=False)


def take_small_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int = 20260619,
    balanced_binary: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic small subset for smoke tests."""

    rng = np.random.default_rng(seed)
    n = int(x.shape[0])
    max_samples = min(max_samples, n)

    if balanced_binary and y.ndim == 1:
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        each = max_samples // 2
        pos_take = min(each, len(positives))
        neg_take = min(max_samples - pos_take, len(negatives))
        indices = np.concatenate(
            [
                rng.choice(positives, size=pos_take, replace=False),
                rng.choice(negatives, size=neg_take, replace=False),
            ]
        )
        if len(indices) < max_samples:
            remaining = np.setdiff1d(np.arange(n), indices, assume_unique=False)
            extra = rng.choice(remaining, size=max_samples - len(indices), replace=False)
            indices = np.concatenate([indices, extra])
    else:
        indices = rng.choice(n, size=max_samples, replace=False)

    rng.shuffle(indices)
    return x[indices], y[indices]


def sequence_one_hot(x: torch.Tensor, num_bases: int = 5) -> torch.Tensor:
    """Convert integer encoded sequences (batch, length) to one-hot (batch, channels, length)."""

    x = x.long().clamp(0, num_bases - 1)
    return torch.nn.functional.one_hot(x, num_classes=num_bases).float().transpose(1, 2)
