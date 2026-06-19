from __future__ import annotations

import torch
from torch import nn

from .data import sequence_one_hot


class SequenceMLPClassifier(nn.Module):
    """Fully connected binary classifier for fixed-length encoded sequences."""

    def __init__(self, sequence_length: int = 1024, num_symbols: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_symbols = num_symbols
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * num_symbols, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = sequence_one_hot(x, self.num_symbols)
        return self.net(one_hot).squeeze(-1)


class SequenceMLPSegmenter(nn.Module):
    """Fully connected base-level segmenter for fixed-length encoded sequences."""

    def __init__(self, sequence_length: int = 1024, num_symbols: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_symbols = num_symbols
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * num_symbols, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, sequence_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = sequence_one_hot(x, self.num_symbols)
        return self.net(one_hot)


class SequenceCNNClassifier(nn.Module):
    """1D CNN binary classifier for encoded DNA windows."""

    def __init__(self, num_symbols: int = 5, channels: int = 64):
        super().__init__()
        self.num_symbols = num_symbols
        self.features = nn.Sequential(
            nn.Conv1d(num_symbols, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(channels, channels * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Linear(channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = sequence_one_hot(x, self.num_symbols)
        features = self.features(one_hot).squeeze(-1)
        return self.classifier(features).squeeze(-1)


class SequenceCNNSegmenter(nn.Module):
    """Fully convolutional base-level segmenter for encoded DNA windows."""

    def __init__(self, num_symbols: int = 5, channels: int = 64):
        super().__init__()
        self.num_symbols = num_symbols
        self.net = nn.Sequential(
            nn.Conv1d(num_symbols, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = sequence_one_hot(x, self.num_symbols)
        return self.net(one_hot).squeeze(1)
