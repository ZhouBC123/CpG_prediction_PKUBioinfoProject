from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    initial_loss: float
    final_loss: float
    output_shape: tuple[int, ...]


def train_torch_smoke(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    steps: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
) -> TrainResult:
    """Run a few optimization steps and return loss/shape diagnostics."""

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    output_shape: tuple[int, ...] | None = None

    for step, (x, y) in enumerate(loader):
        if step >= steps:
            break
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        output_shape = tuple(logits.shape)
        loss = loss_fn(logits, y.float())
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss during smoke training")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if not losses or output_shape is None:
        raise RuntimeError("no batches were processed")

    return TrainResult(initial_loss=losses[0], final_loss=losses[-1], output_shape=output_shape)
