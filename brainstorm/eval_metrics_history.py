"""Utilities for persisting per-epoch evaluation metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _to_serializable(value: Any) -> Any:
    """Convert common numeric tensor/array scalar values to JSON/CSV-friendly values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def append_epoch_metrics_history(
    save_dir: str | Path,
    row: Mapping[str, Any],
    *,
    reset: bool = False,
) -> tuple[Path, Path]:
    """Append an epoch metrics row to JSONL and CSV files.

    The CSV writer preserves existing columns and expands the header if a later row
    introduces new metrics.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "metrics_history.csv"
    jsonl_path = save_dir / "metrics_history.jsonl"

    if reset:
        csv_path.unlink(missing_ok=True)
        jsonl_path.unlink(missing_ok=True)

    serializable_row = {str(key): _to_serializable(value) for key, value in row.items()}

    with jsonl_path.open("a") as f:
        f.write(json.dumps(serializable_row, sort_keys=True) + "\n")

    existing_rows: list[dict[str, Any]] = []
    fieldnames = list(serializable_row.keys())

    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []
            fieldnames = existing_fieldnames + [
                key for key in serializable_row.keys() if key not in existing_fieldnames
            ]

    existing_rows.append(serializable_row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)

    return csv_path, jsonl_path


def resolve_checkpoint_dir(logging_cfg: Any) -> Path:
    """Return the directory used for model checkpoints.

    Older configs only define ``logging.save_dir``. Keep that as a fallback so
    existing Hydra runs and ad-hoc configs continue to work.
    """
    checkpoint_dir = logging_cfg.get("checkpoint_dir", None)
    if checkpoint_dir:
        return Path(checkpoint_dir)
    return Path(logging_cfg.save_dir)
