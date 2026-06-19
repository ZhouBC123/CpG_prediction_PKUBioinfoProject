#!/usr/bin/env python
"""Build CpG island binary-classification and segmentation datasets.

Inputs are UCSC hg38 cpgIslandExt annotations and hg38.fa.gz.
Outputs are compressed NumPy archives under processed_data/.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
VAL_CHROMS = {"chr8", "chr9"}
TEST_CHROMS = {"chr10", "chr11"}
ENCODING = {"A": 0, "C": 1, "G": 2, "T": 3, "N/other": 4}


def split_for_chrom(chrom: str) -> str:
    if chrom in VAL_CHROMS:
        return "val"
    if chrom in TEST_CHROMS:
        return "test"
    return "train"


def read_cpg_table(path: Path) -> pd.DataFrame:
    columns = [
        "bin",
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "length",
        "cpgNum",
        "gcNum",
        "perCpg",
        "perGc",
        "obsExp",
    ]
    df = pd.read_csv(path, sep="\t", header=None, names=columns, compression="gzip")
    df = df[df["chrom"].isin(STANDARD_CHROMS)].copy()
    df["split"] = df["chrom"].map(split_for_chrom)
    df = df.sort_values(["chrom", "chromStart", "chromEnd"]).reset_index(drop=True)
    return df


def read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            chrom, size = line.rstrip("\n").split("\t")[:2]
            if chrom in STANDARD_CHROMS:
                sizes[chrom] = int(size)
    return sizes


def iter_fasta(path: Path, keep_chroms: set[str]) -> Iterable[tuple[str, str]]:
    chrom = None
    chunks: list[str] = []

    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if chrom in keep_chroms:
                    yield chrom, "".join(chunks).upper()
                chrom = line[1:].split()[0]
                chunks = []
            elif chrom in keep_chroms:
                chunks.append(line)

    if chrom in keep_chroms:
        yield chrom, "".join(chunks).upper()


def make_encoder() -> np.ndarray:
    table = np.full(256, 4, dtype=np.uint8)
    for base, code in [("A", 0), ("C", 1), ("G", 2), ("T", 3), ("N", 4)]:
        table[ord(base)] = code
        table[ord(base.lower())] = code
    return table


def center_window(start: int, end: int, chrom_len: int, window_size: int) -> tuple[int, int]:
    center = (start + end) // 2
    win_start = center - window_size // 2
    win_start = max(0, min(win_start, chrom_len - window_size))
    return win_start, win_start + window_size


def window_sum(cumsum: np.ndarray, start: int, end: int) -> int:
    return int(cumsum[end] - cumsum[start])


def sample_negative_windows(
    rng: np.random.Generator,
    chrom_len: int,
    window_size: int,
    count: int,
    label_cumsum: np.ndarray,
    n_cumsum: np.ndarray,
    max_n: int,
    max_attempts_per_window: int,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(count * max_attempts_per_window, 1000)

    while len(windows) < count and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, chrom_len - window_size + 1))
        end = start + window_size
        if window_sum(label_cumsum, start, end) != 0:
            continue
        if window_sum(n_cumsum, start, end) > max_n:
            continue
        windows.append((start, end))

    if len(windows) != count:
        raise RuntimeError(
            f"sampled {len(windows)}/{count} negative windows after {attempts} attempts"
        )
    return windows


def append_index(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample_id", "chrom", "start", "end", "label", "source"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def save_split_npz(
    out_path: Path,
    x_chunks: list[np.ndarray],
    y_chunks: list[np.ndarray],
    starts: list[int],
    ends: list[int],
    chroms: list[str],
    sources: list[str],
) -> dict[str, object]:
    if x_chunks:
        x = np.stack(x_chunks).astype(np.uint8, copy=False)
        y = np.stack(y_chunks).astype(np.uint8, copy=False)
    else:
        x = np.empty((0, 0), dtype=np.uint8)
        y = np.empty((0,), dtype=np.uint8)

    np.savez_compressed(
        out_path,
        X=x,
        y=y,
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        chroms=np.asarray(chroms),
        sources=np.asarray(sources),
    )

    return {
        "samples": int(x.shape[0]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "positive_samples": int(np.asarray(y).sum()) if y.ndim == 1 else None,
        "positive_bases": int(np.asarray(y).sum()) if y.ndim == 2 else None,
        "file": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/root/autodl-tmp/bioinfo_data/raw/ucsc_hg38"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-n-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    binary_dir = out_dir / "binary"
    segmentation_dir = out_dir / "segmentation"
    for directory in [out_dir, binary_dir, segmentation_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    cpg_path = raw_dir / "cpgIslandExt.txt.gz"
    fasta_path = raw_dir / "hg38.fa.gz"
    sizes_path = raw_dir / "hg38.chrom.sizes"
    for path in [cpg_path, fasta_path, sizes_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    rng = np.random.default_rng(args.seed)
    encoder = make_encoder()
    max_n = int(args.window_size * args.max_n_fraction)

    cpg = read_cpg_table(cpg_path)
    chrom_sizes = read_chrom_sizes(sizes_path)
    cpg.to_csv(out_dir / "cpg_islands_standard.tsv", sep="\t", index=False)

    per_task = {
        "binary": {split: defaultdict(list) for split in ["train", "val", "test"]},
        "segmentation": {split: defaultdict(list) for split in ["train", "val", "test"]},
    }
    index_rows = {
        "binary": {split: [] for split in ["train", "val", "test"]},
        "segmentation": {split: [] for split in ["train", "val", "test"]},
    }

    for chrom, seq in iter_fasta(fasta_path, set(STANDARD_CHROMS)):
        if len(seq) < args.window_size:
            continue
        if chrom_sizes.get(chrom) and chrom_sizes[chrom] != len(seq):
            raise ValueError(f"{chrom} length mismatch: fasta={len(seq)} sizes={chrom_sizes[chrom]}")

        chrom_intervals = cpg[cpg["chrom"] == chrom]
        if chrom_intervals.empty:
            continue

        encoded = encoder[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        n_mask = encoded == 4
        cpg_mask = np.zeros(len(seq), dtype=np.bool_)
        for row in chrom_intervals.itertuples(index=False):
            cpg_mask[int(row.chromStart) : int(row.chromEnd)] = True

        label_cumsum = np.concatenate(([0], np.cumsum(cpg_mask, dtype=np.int64)))
        n_cumsum = np.concatenate(([0], np.cumsum(n_mask, dtype=np.int64)))

        positive_windows: list[tuple[int, int]] = []
        for row in chrom_intervals.itertuples(index=False):
            start, end = center_window(int(row.chromStart), int(row.chromEnd), len(seq), args.window_size)
            if window_sum(n_cumsum, start, end) <= max_n:
                positive_windows.append((start, end))

        negative_count = int(round(len(positive_windows) * args.negative_ratio))
        negative_windows = sample_negative_windows(
            rng=rng,
            chrom_len=len(seq),
            window_size=args.window_size,
            count=negative_count,
            label_cumsum=label_cumsum,
            n_cumsum=n_cumsum,
            max_n=max_n,
            max_attempts_per_window=200,
        )

        split = split_for_chrom(chrom)
        all_windows = [(start, end, 1, "cpg_centered") for start, end in positive_windows]
        all_windows.extend((start, end, 0, "non_cpg_random") for start, end in negative_windows)
        rng.shuffle(all_windows)

        for start, end, label, source in all_windows:
            seq_window = encoded[start:end].copy()
            seg_label = cpg_mask[start:end].astype(np.uint8, copy=True)

            sample_id = len(index_rows["binary"][split])
            for task in ["binary", "segmentation"]:
                store = per_task[task][split]
                store["X"].append(seq_window)
                store["starts"].append(start)
                store["ends"].append(end)
                store["chroms"].append(chrom)
                store["sources"].append(source)

                if task == "binary":
                    store["y"].append(np.asarray(label, dtype=np.uint8))
                else:
                    store["y"].append(seg_label)

                index_rows[task][split].append(
                    {
                        "sample_id": sample_id,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "label": label,
                        "source": source,
                    }
                )

        print(
            f"{chrom}: split={split} positives={len(positive_windows)} "
            f"negatives={len(negative_windows)}"
        )

    manifest: dict[str, object] = {
        "raw_dir": str(raw_dir),
        "output_dir": str(out_dir),
        "window_size": args.window_size,
        "negative_ratio": args.negative_ratio,
        "max_n_fraction": args.max_n_fraction,
        "seed": args.seed,
        "encoding": ENCODING,
        "splits": {
            "train": [chrom for chrom in STANDARD_CHROMS if split_for_chrom(chrom) == "train"],
            "val": sorted(VAL_CHROMS),
            "test": sorted(TEST_CHROMS),
        },
        "source_files": {
            "cpg_islands": str(cpg_path),
            "fasta": str(fasta_path),
            "chrom_sizes": str(sizes_path),
        },
        "tasks": {"binary": {}, "segmentation": {}},
    }

    for task, task_dir in [("binary", binary_dir), ("segmentation", segmentation_dir)]:
        for split in ["train", "val", "test"]:
            store = per_task[task][split]
            npz_path = task_dir / f"{split}.npz"
            index_path = task_dir / f"{split}_index.tsv"
            stats = save_split_npz(
                npz_path,
                store["X"],
                store["y"],
                store["starts"],
                store["ends"],
                store["chroms"],
                store["sources"],
            )
            append_index(index_path, index_rows[task][split])
            stats["index_file"] = str(index_path)
            manifest["tasks"][task][split] = stats

    readme = out_dir / "README.md"
    readme.write_text(
        "# Processed CpG Datasets\n\n"
        "Encoding: A=0, C=1, G=2, T=3, N/other=4.\n\n"
        "- `binary/{train,val,test}.npz`: `X` has shape `(samples, 1024)`, `y` has shape `(samples,)`.\n"
        "- `segmentation/{train,val,test}.npz`: `X` has shape `(samples, 1024)`, `y` has shape `(samples, 1024)`.\n"
        "- `*_index.tsv` files record chromosome coordinates and sample source.\n"
        "- Chromosome split is train: all standard chromosomes except chr8/chr9/chr10/chr11, val: chr8/chr9, test: chr10/chr11.\n",
        encoding="utf-8",
    )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["tasks"], indent=2))


if __name__ == "__main__":
    main()
