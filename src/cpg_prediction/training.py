from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        optimizer.step()
        batch = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * batch
        total_items += batch
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    logits_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        logits = model(x)
        loss = loss_fn(logits, y)
        batch = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * batch
        total_items += batch
        logits_chunks.append(logits.detach().cpu())
        y_chunks.append(y.detach().cpu())

    return torch.cat(logits_chunks), torch.cat(y_chunks), total_loss / max(total_items, 1)


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
