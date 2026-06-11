#!/usr/bin/env python3
"""Scan configured EEG datasets and report maximum EEG channel dimensions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brainstorm.data.eeg_word_aligned_dataset import (  # noqa: E402
    scan_bids_eeg_channel_counts,
    scan_zuco_channel_counts,
)


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _entry_get(entry: Any, key: str, default: Any = None) -> Any:
    return entry.get(key, default) if hasattr(entry, "get") else default


def scan_entry(entry: Any) -> tuple[str, int | None, int]:
    dataset_type = str(_entry_get(entry, "dataset_type", ""))
    dataset_name = str(_entry_get(entry, "dataset_name", dataset_type))
    root = _entry_get(entry, "root")
    tasks = _as_list(_entry_get(entry, "tasks", None))
    if root is None:
        return dataset_name, None, 0

    if dataset_type == "zuco":
        counts = scan_zuco_channel_counts(root)
    elif dataset_type in {"eegdash", "openneuro_eeg", "openneuro_ds004408", "openneuro_ds007808"}:
        counts = scan_bids_eeg_channel_counts(root, tasks=tasks)
    else:
        return dataset_name, None, 0

    if not counts:
        return dataset_name, None, 0
    return dataset_name, max(item.n_channels for item in counts), len(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval_criss_cross_word_classification_eeg_reading.yaml"),
        help="EEG evaluation YAML to scan.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    entries = list(cfg.data.get("datasets") or [cfg.data])
    pooled_max: int | None = None

    print(f"Config: {args.config}")
    print("dataset\trecordings\tmax_channels")
    for entry in entries:
        name, max_channels, n_recordings = scan_entry(entry)
        max_text = str(max_channels) if max_channels is not None else "unavailable"
        print(f"{name}\t{n_recordings}\t{max_text}")
        if max_channels is not None:
            pooled_max = max(max_channels, pooled_max or 0)

    if pooled_max is None:
        print("No readable EEG channel counts found.", file=sys.stderr)
        return 1

    print(f"\npooled_max_channel_dim\t{pooled_max}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
