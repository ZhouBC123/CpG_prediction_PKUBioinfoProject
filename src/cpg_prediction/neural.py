from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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


class _ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout1d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceUNetSegmenter(nn.Module):
    """1D U-Net base-level segmenter for encoded DNA windows."""

    def __init__(
        self,
        num_symbols: int = 5,
        base_channels: int = 32,
        depth: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.num_symbols = num_symbols
        encoder_channels = [base_channels * (2**i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = num_symbols
        for channels in encoder_channels:
            self.down_blocks.append(_ConvBlock1d(in_channels, channels, dropout=dropout))
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            in_channels = channels

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = _ConvBlock1d(encoder_channels[-1], bottleneck_channels, dropout=dropout)

        self.up_transposes = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        decoder_in = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.up_transposes.append(nn.ConvTranspose1d(decoder_in, skip_channels, kernel_size=2, stride=2))
            self.up_blocks.append(_ConvBlock1d(skip_channels * 2, skip_channels, dropout=dropout))
            decoder_in = skip_channels

        self.output = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = sequence_one_hot(x, self.num_symbols)
        skips = []

        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.up_transposes, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.output(x).squeeze(1)
