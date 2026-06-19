#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cpg_prediction.data import load_npz_arrays, take_small_subset  # noqa: E402
from cpg_prediction.metrics import binary_metrics, segmentation_metrics  # noqa: E402
from cpg_prediction.neural import (  # noqa: E402
    SequenceCNNClassifier,
    SequenceCNNSegmenter,
    SequenceMLPClassifier,
    SequenceMLPSegmenter,
)
from cpg_prediction.probabilistic import MarkovBinaryClassifier, SupervisedHMMCpGSegmenter  # noqa: E402
from cpg_prediction.training import predict_logits, save_checkpoint, train_epoch  # noqa: E402


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def sigmoid_numpy(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).numpy()


def select_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    seed: int,
    balanced_binary: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples <= 0 or max_samples >= x.shape[0]:
        return x, y
    return take_small_subset(x, y, max_samples=max_samples, seed=seed, balanced_binary=balanced_binary)


def make_neural_specs(binary_epochs: int, segmentation_epochs: int) -> list[dict[str, Any]]:
    return [
        {
            "family": "mlp_binary",
            "name": "mlp_binary_h128_lr1e-3",
            "task": "binary",
            "batch_size": 128,
            "make_model": lambda: SequenceMLPClassifier(hidden_dim=128),
            "config": {"hidden_dim": 128, "lr": 1e-3, "epochs": binary_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "mlp_binary",
            "name": "mlp_binary_h256_lr5e-4",
            "task": "binary",
            "batch_size": 128,
            "make_model": lambda: SequenceMLPClassifier(hidden_dim=256),
            "config": {"hidden_dim": 256, "lr": 5e-4, "epochs": binary_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "cnn_binary",
            "name": "cnn_binary_c32_lr1e-3",
            "task": "binary",
            "batch_size": 128,
            "make_model": lambda: SequenceCNNClassifier(channels=32),
            "config": {"channels": 32, "lr": 1e-3, "epochs": binary_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "cnn_binary",
            "name": "cnn_binary_c64_lr5e-4",
            "task": "binary",
            "batch_size": 128,
            "make_model": lambda: SequenceCNNClassifier(channels=64),
            "config": {"channels": 64, "lr": 5e-4, "epochs": binary_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "mlp_segmentation",
            "name": "mlp_segmentation_h128_lr1e-3",
            "task": "segmentation",
            "batch_size": 64,
            "make_model": lambda: SequenceMLPSegmenter(hidden_dim=128),
            "config": {"hidden_dim": 128, "lr": 1e-3, "epochs": segmentation_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "mlp_segmentation",
            "name": "mlp_segmentation_h256_lr5e-4",
            "task": "segmentation",
            "batch_size": 64,
            "make_model": lambda: SequenceMLPSegmenter(hidden_dim=256),
            "config": {"hidden_dim": 256, "lr": 5e-4, "epochs": segmentation_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "cnn_segmentation",
            "name": "cnn_segmentation_c32_lr1e-3",
            "task": "segmentation",
            "batch_size": 64,
            "make_model": lambda: SequenceCNNSegmenter(channels=32),
            "config": {"channels": 32, "lr": 1e-3, "epochs": segmentation_epochs, "weight_decay": 1e-5},
        },
        {
            "family": "cnn_segmentation",
            "name": "cnn_segmentation_c64_lr5e-4",
            "task": "segmentation",
            "batch_size": 64,
            "make_model": lambda: SequenceCNNSegmenter(channels=64),
            "config": {"channels": 64, "lr": 5e-4, "epochs": segmentation_epochs, "weight_decay": 1e-5},
        },
    ]


def load_neural_arrays(
    processed_dir: Path,
    max_train_binary: int | None,
    max_train_segmentation: int | None,
    seed: int,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    x_train_b, y_train_b = load_npz_arrays(processed_dir, "binary", "train")
    x_val_b, y_val_b = load_npz_arrays(processed_dir, "binary", "val")
    x_test_b, y_test_b = load_npz_arrays(processed_dir, "binary", "test")
    x_train_s, y_train_s = load_npz_arrays(processed_dir, "segmentation", "train")
    x_val_s, y_val_s = load_npz_arrays(processed_dir, "segmentation", "val")
    x_test_s, y_test_s = load_npz_arrays(processed_dir, "segmentation", "test")

    x_train_b, y_train_b = select_subset(x_train_b, y_train_b, max_train_binary, seed, balanced_binary=True)
    x_train_s, y_train_s = select_subset(x_train_s, y_train_s, max_train_segmentation, seed, balanced_binary=False)

    return {
        "binary": {
            "train": (x_train_b, y_train_b),
            "val": (x_val_b, y_val_b),
            "test": (x_test_b, y_test_b),
        },
        "segmentation": {
            "train": (x_train_s, y_train_s),
            "val": (x_val_s, y_val_s),
            "test": (x_test_s, y_test_s),
        },
    }


def make_task_loaders(arrays: dict[str, tuple[np.ndarray, np.ndarray]], batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    return (
        make_loader(x_train, y_train, batch_size, True),
        make_loader(x_val, y_val, batch_size * 2, False),
        make_loader(x_test, y_test, batch_size * 2, False),
    )


def metric_key_for_task(task: str) -> str:
    return "f1" if task == "binary" else "base_f1"


def evaluate_markov(
    processed_dir: Path,
    alphas: list[float],
    max_train_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    x_train, y_train = load_npz_arrays(processed_dir, "binary", "train")
    x_val, y_val = load_npz_arrays(processed_dir, "binary", "val")
    x_test, y_test = load_npz_arrays(processed_dir, "binary", "test")
    x_train, y_train = select_subset(x_train, y_train, max_train_samples, seed, balanced_binary=True)

    candidates = []
    for alpha in alphas:
        start = time.time()
        model = MarkovBinaryClassifier(alpha=alpha).fit(x_train, y_train)
        val_score = model.predict_proba(x_val)[:, 1]
        test_score = model.predict_proba(x_test)[:, 1]
        val_metrics = binary_metrics(y_val, val_score)
        test_metrics = binary_metrics(y_test, test_score)
        candidates.append(
            {
                "alpha": alpha,
                "train_samples": int(x_train.shape[0]),
                "val": val_metrics,
                "test": test_metrics,
                "seconds": time.time() - start,
            }
        )

    best = max(candidates, key=lambda item: item["val"]["f1"])
    return {"task": "binary", "model": "markov", "candidates": candidates, "best": best}


def evaluate_hmm(
    processed_dir: Path,
    alphas: list[float],
    max_train_samples: int | None,
    max_eval_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    x_train, y_train = load_npz_arrays(processed_dir, "segmentation", "train")
    x_val, y_val = load_npz_arrays(processed_dir, "segmentation", "val")
    x_test, y_test = load_npz_arrays(processed_dir, "segmentation", "test")
    x_train, y_train = select_subset(x_train, y_train, max_train_samples, seed, balanced_binary=False)
    x_val_eval, y_val_eval = select_subset(x_val, y_val, max_eval_samples, seed + 1, balanced_binary=False)
    x_test_eval, y_test_eval = select_subset(x_test, y_test, max_eval_samples, seed + 2, balanced_binary=False)

    candidates = []
    for alpha in alphas:
        start = time.time()
        model = SupervisedHMMCpGSegmenter(alpha=alpha).fit(x_train, y_train)
        val_pred = model.predict(x_val_eval)
        test_pred = model.predict(x_test_eval)
        val_metrics = segmentation_metrics(y_val_eval, val_pred)
        test_metrics = segmentation_metrics(y_test_eval, test_pred)
        candidates.append(
            {
                "alpha": alpha,
                "train_samples": int(x_train.shape[0]),
                "eval_samples": int(x_val_eval.shape[0]),
                "test_eval_samples": int(x_test_eval.shape[0]),
                "val": val_metrics,
                "test": test_metrics,
                "seconds": time.time() - start,
            }
        )

    best = max(candidates, key=lambda item: item["val"]["base_f1"])
    return {"task": "segmentation", "model": "hmm", "candidates": candidates, "best": best}


def train_neural_config(
    name: str,
    family: str,
    phase: str,
    task: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict[str, Any],
    train_samples: int,
    device: str,
    model_dir: Path,
) -> dict[str, Any]:
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 0.0)))
    epochs = int(config["epochs"])
    history = []
    best_state = None
    best_val_key = -1.0
    best_epoch = -1
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_logits, val_y, val_loss = predict_logits(model, val_loader, device)
        val_score = sigmoid_numpy(val_logits)
        y_val_np = val_y.numpy()
        if task == "binary":
            metrics = binary_metrics(y_val_np, val_score)
        else:
            metrics = segmentation_metrics(y_val_np, val_score)
        key = float(metrics[metric_key_for_task(task)])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val": metrics})
        if key > best_val_key:
            best_val_key = key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_logits, test_y, test_loss = predict_logits(model, test_loader, device)
    test_score = sigmoid_numpy(test_logits)
    test_y_np = test_y.numpy()
    if task == "binary":
        test_metrics = binary_metrics(test_y_np, test_score)
    else:
        test_metrics = segmentation_metrics(test_y_np, test_score)

    checkpoint = model_dir / phase / f"{name}.pt"
    save_checkpoint(
        model,
        checkpoint,
        {
            "name": name,
            "family": family,
            "phase": phase,
            "task": task,
            "config": config,
            "train_samples": train_samples,
            "best_epoch": best_epoch,
        },
    )
    return {
        "name": name,
        "family": family,
        "phase": phase,
        "task": task,
        "config": config,
        "train_samples": train_samples,
        "best_epoch": best_epoch,
        "history": history,
        "test_loss": test_loss,
        "test": test_metrics,
        "checkpoint": str(checkpoint),
        "seconds": time.time() - start,
    }


def evaluate_neural(
    processed_dir: Path,
    model_dir: Path,
    phase: str,
    specs: list[dict[str, Any]],
    max_train_binary: int | None,
    max_train_segmentation: int | None,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    arrays = load_neural_arrays(processed_dir, max_train_binary, max_train_segmentation, seed)
    loaders_by_task: dict[tuple[str, int], tuple[DataLoader, DataLoader, DataLoader]] = {}
    results = []

    for index, spec in enumerate(specs):
        task = str(spec["task"])
        batch_size = int(spec["batch_size"])
        loader_key = (task, batch_size)
        if loader_key not in loaders_by_task:
            loaders_by_task[loader_key] = make_task_loaders(arrays[task], batch_size)
        train_loader, val_loader, test_loader = loaders_by_task[loader_key]
        train_samples = int(arrays[task]["train"][0].shape[0])

        torch.manual_seed(seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + index)

        print(f"training {phase}/{spec['name']} on {train_samples} samples", flush=True)
        result = train_neural_config(
            name=str(spec["name"]),
            family=str(spec["family"]),
            phase=phase,
            task=task,
            model=spec["make_model"](),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=dict(spec["config"]),
            train_samples=train_samples,
            device=device,
            model_dir=model_dir,
        )
        print(f"done {phase}/{spec['name']}: test={result['test']}", flush=True)
        results.append(result)

    return results


def best_neural_by_family(results: list[dict[str, Any]], family: str) -> dict[str, Any]:
    candidates = [item for item in results if item["family"] == family]
    if not candidates:
        raise RuntimeError(f"no neural candidates for {family}")
    key = metric_key_for_task(str(candidates[0]["task"]))
    return max(candidates, key=lambda item: item["history"][item["best_epoch"] - 1]["val"][key])


def selected_full_specs(
    tuning_results: list[dict[str, Any]],
    full_binary_epochs: int,
    full_segmentation_epochs: int,
) -> list[dict[str, Any]]:
    full_specs_by_name = {
        spec["name"]: spec for spec in make_neural_specs(full_binary_epochs, full_segmentation_epochs)
    }
    selected = []
    for family in ["mlp_binary", "cnn_binary", "mlp_segmentation", "cnn_segmentation"]:
        best = best_neural_by_family(tuning_results, family)
        selected.append(full_specs_by_name[best["name"]])
    return selected


def neural_summary_row(result: dict[str, Any], label: str, config_note: str | None = None) -> dict[str, Any]:
    key = metric_key_for_task(str(result["task"]))
    val_metrics = result["history"][result["best_epoch"] - 1]["val"]
    test_metrics = result["test"]
    if result["task"] == "binary":
        other = f"test AUROC={test_metrics['auroc']:.4f}"
    else:
        other = f"test IoU={test_metrics['base_iou']:.4f}"
    config = str(result["name"])
    if config_note:
        config = f"{config}; {config_note}"
    return {
        "task": result["task"],
        "model": label,
        "phase": result["phase"],
        "train_samples": result["train_samples"],
        "config": config,
        "val_f1": val_metrics[key],
        "test_f1": test_metrics[key],
        "other": other,
    }


def write_results_markdown(results: dict[str, Any], path: Path) -> None:
    lines = [
        "# Experiment Results",
        "",
        "This file is generated from `results/metrics.json` after running `scripts/run_experiments.py`.",
        "",
        "## Setup",
        "",
        f"- Device: `{results['device']}`",
        f"- Seed: `{results['seed']}`",
        f"- Processed data: `{results['processed_dir']}`",
        f"- Neural tuning limits: binary `{results['limits']['tune_train_binary']}`, segmentation `{results['limits']['tune_train_segmentation']}`",
        "- Full neural training uses the complete train split after selecting the best tuning config per model family.",
        "",
        "## Best Validation Choices",
        "",
        "| Task | Model | Phase | Train samples | Selected config | Validation F1 | Test F1 | Other test metrics |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]

    for item in results["summary_rows"]:
        lines.append(
            f"| {item['task']} | {item['model']} | {item['phase']} | {item['train_samples']} | `{item['config']}` | "
            f"{item['val_f1']:.4f} | {item['test_f1']:.4f} | {item['other']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Markov and HMM baselines are fit with Laplace smoothing grid search over alpha values.",
            "- Neural candidates are first checked on deterministic smaller training subsets, then the best MLP/CNN candidate per task is retrained on the full training split.",
            "- Checkpoints are written under `models/` and intentionally ignored by git.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--tune-train-binary", type=int, default=12000)
    parser.add_argument("--tune-train-segmentation", type=int, default=8000)
    parser.add_argument("--tune-binary-epochs", type=int, default=5)
    parser.add_argument("--tune-segmentation-epochs", type=int, default=4)
    parser.add_argument("--full-binary-epochs", type=int, default=8)
    parser.add_argument("--full-segmentation-epochs", type=int, default=6)
    parser.add_argument("--max-markov-train", type=int, default=0)
    parser.add_argument("--max-hmm-train", type=int, default=0)
    parser.add_argument("--max-hmm-eval", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    print("evaluating Markov baseline", flush=True)
    markov = evaluate_markov(args.processed_dir, [0.1, 1.0, 10.0], args.max_markov_train, args.seed)
    print("evaluating HMM baseline", flush=True)
    hmm = evaluate_hmm(args.processed_dir, [0.1, 1.0, 10.0], args.max_hmm_train, args.max_hmm_eval, args.seed)

    tuning_specs = make_neural_specs(args.tune_binary_epochs, args.tune_segmentation_epochs)
    neural_tuning = evaluate_neural(
        args.processed_dir,
        args.model_dir,
        "tuning",
        tuning_specs,
        args.tune_train_binary,
        args.tune_train_segmentation,
        args.seed,
        device,
    )

    full_specs = selected_full_specs(neural_tuning, args.full_binary_epochs, args.full_segmentation_epochs)
    neural_full = evaluate_neural(
        args.processed_dir,
        args.model_dir,
        "full",
        full_specs,
        0,
        0,
        args.seed + 1000,
        device,
    )

    summary_rows = []
    best_markov = markov["best"]
    summary_rows.append(
        {
            "task": "binary",
            "model": "Markov",
            "phase": "full",
            "train_samples": best_markov["train_samples"],
            "config": f"alpha={best_markov['alpha']}",
            "val_f1": best_markov["val"]["f1"],
            "test_f1": best_markov["test"]["f1"],
            "other": f"test AUROC={best_markov['test']['auroc']:.4f}",
        }
    )
    best_hmm = hmm["best"]
    summary_rows.append(
        {
            "task": "segmentation",
            "model": "HMM",
            "phase": "full",
            "train_samples": best_hmm["train_samples"],
            "config": f"alpha={best_hmm['alpha']}",
            "val_f1": best_hmm["val"]["base_f1"],
            "test_f1": best_hmm["test"]["base_f1"],
            "other": f"test IoU={best_hmm['test']['base_iou']:.4f}",
        }
    )

    labels = {
        "mlp_binary": "MLP",
        "cnn_binary": "CNN",
        "mlp_segmentation": "MLP",
        "cnn_segmentation": "CNN",
    }
    tuning_by_family = {family: best_neural_by_family(neural_tuning, family) for family in labels}
    for family, label in labels.items():
        full_best = best_neural_by_family(neural_full, family)
        tuned = tuning_by_family[family]
        note = f"selected from tuning {tuned['name']}"
        summary_rows.append(neural_summary_row(full_best, label, note))

    results = {
        "seed": args.seed,
        "device": device,
        "processed_dir": str(args.processed_dir),
        "limits": {
            "tune_train_binary": args.tune_train_binary,
            "tune_train_segmentation": args.tune_train_segmentation,
            "tune_binary_epochs": args.tune_binary_epochs,
            "tune_segmentation_epochs": args.tune_segmentation_epochs,
            "full_binary_epochs": args.full_binary_epochs,
            "full_segmentation_epochs": args.full_segmentation_epochs,
            "max_markov_train": args.max_markov_train,
            "max_hmm_train": args.max_hmm_train,
            "max_hmm_eval": args.max_hmm_eval,
        },
        "markov": markov,
        "hmm": hmm,
        "neural_tuning": neural_tuning,
        "neural_full": neural_full,
        "summary_rows": summary_rows,
    }

    metrics_path = args.results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_results_markdown(results, Path("docs/results.md"))
    print(json.dumps({"summary_rows": summary_rows, "metrics_path": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
