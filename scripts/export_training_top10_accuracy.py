#!/usr/bin/env python3
"""Export top-10 training curves from curriculum and ds004408 fine-tuning logs.

The script reads per-epoch metric histories saved as either ``metrics_history``
or ``epoch_metrics`` files, extracts top-k retrieval accuracy columns, writes a
long CSV plus per-figure pivot CSVs, and optionally renders PNG/PDF plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


HISTORY_NAMES = (
    "metrics_history.csv",
    "epoch_metrics.csv",
    "metrics_history.jsonl",
    "epoch_metrics.jsonl",
)

METRIC_RE = re.compile(
    r"^(?:(?P<split>[^/]+)/)?"
    r"(?P<balanced>balanced_)?top(?P<top_k>\d+)_accuracy_retrieval(?P<size>\d+)$"
)

PRETRAIN_SKIP_SUFFIXES = ("_step", "/step")


@dataclass(frozen=True)
class RunInfo:
    source: str
    run_id: str
    model: str
    stage: str
    run_dir: Path
    history_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract top-10 accuracy curves saved during training and export "
            "CSV/PNG/PDF artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/training_top10_accuracy"),
        help="Directory for exported CSVs, figures and manifest.",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Root to scan recursively for metric histories. Defaults to the "
            "curriculum and ds004408 log roots when present."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Explicit run directory containing metrics_history.* or epoch_metrics.*.",
    )
    parser.add_argument(
        "--include-pattern",
        default=r"(eeg_language_curriculum_three_models|word_classification_ds004408)",
        help="Regex applied to history-file paths found under scan roots.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k value to extract. Use 0 to export all top-k values found.",
    )
    parser.add_argument(
        "--retrieval-sizes",
        type=int,
        nargs="*",
        default=[50, 250],
        help="Retrieval vocabulary sizes to export. Omit values with '--retrieval-sizes' to export all found.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["val", "test_during_train", "test_at_best_val"],
        help="Metric prefixes/splits to export. Omit values with '--splits' to export all found.",
    )
    parser.add_argument(
        "--metric-variants",
        nargs="*",
        choices=["balanced", "raw"],
        default=["balanced", "raw"],
        help="Export balanced, raw top-k accuracy, or both.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only write CSV/JSON outputs.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_history(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    return read_csv(path)


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def to_intish(value: Any) -> str:
    number = to_float(value)
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return "" if value is None else str(value)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_history_file(run_dir: Path) -> Path | None:
    for name in HISTORY_NAMES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def default_scan_roots() -> list[Path]:
    candidates = [
        Path("logs/eeg_language_curriculum_three_models"),
        Path("logs/word_classification_ds004408_four_way"),
        Path("logs/word_classification_ds004408_eeg"),
        Path("logs/word_classification_ds004408_three_way"),
    ]
    return [path for path in candidates if path.exists()]


def discover_history_files(
    scan_roots: Sequence[Path],
    run_dirs: Sequence[Path],
    include_pattern: str,
) -> list[Path]:
    include_re = re.compile(include_pattern) if include_pattern else None
    files: dict[Path, None] = {}
    run_directories: dict[Path, None] = {}

    for run_dir in run_dirs:
        history = find_history_file(run_dir)
        if history is not None:
            files[history.resolve()] = None

    for root in scan_roots:
        if not root.exists():
            continue
        for name in HISTORY_NAMES:
            for path in root.rglob(name):
                text = path.as_posix()
                if include_re is not None and not include_re.search(text):
                    continue
                run_directories[path.parent.resolve()] = None

    for run_dir in sorted(run_directories):
        history = find_history_file(run_dir)
        if history is not None:
            files[history.resolve()] = None

    return sorted(files.keys())


def part_after(parts: Sequence[str], marker: str, offset: int = 1) -> str:
    try:
        index = parts.index(marker)
    except ValueError:
        return ""
    target = index + offset
    return parts[target] if 0 <= target < len(parts) else ""


def infer_run_info(history_file: Path) -> RunInfo:
    run_dir = history_file.parent
    parts = history_file.parts
    source = "other"
    run_id = ""
    model = run_dir.name
    stage = ""

    if "eeg_language_curriculum_three_models" in parts:
        source = "curriculum"
        run_id = part_after(parts, "eeg_language_curriculum_three_models", 1)
        experiment = run_dir.name
        match = re.match(r"^eeg_curriculum_(?P<label>.+)_(?P<stage>reading|language)_seed\d+$", experiment)
        if match:
            model = match.group("label")
            stage = match.group("stage")
        else:
            model = experiment
    elif "word_classification_ds004408_four_way" in parts:
        source = "ds004408_four_way"
        run_id = part_after(parts, "word_classification_ds004408_four_way", 1)
        model = part_after(parts, "word_classification_ds004408_four_way", 2) or run_dir.name
        stage = "finetuning"
    elif "word_classification_ds004408_eeg" in parts:
        source = "ds004408_three_way"
        run_id = part_after(parts, "word_classification_ds004408_eeg", 1)
        model = part_after(parts, "word_classification_ds004408_eeg", 2) or run_dir.name
        stage = "finetuning"
    elif "word_classification_ds004408_three_way" in parts:
        source = "ds004408_three_way"
        run_id = part_after(parts, "word_classification_ds004408_three_way", 1)
        model = part_after(parts, "word_classification_ds004408_three_way", 2) or run_dir.name
        stage = "finetuning"

    return RunInfo(
        source=source,
        run_id=run_id or "unknown_run",
        model=model,
        stage=stage,
        run_dir=run_dir,
        history_file=history_file,
    )


def row_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def extract_records(
    info: RunInfo,
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    retrieval_sizes: set[int] | None,
    splits: set[str] | None,
    metric_variants: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(rows, start=1):
        epoch = row_value(row, "epoch", "trainer/epoch")
        global_step = row_value(row, "global_step", "step", "trainer/global_step")
        train_loss = row_value(row, "train/loss", "train_loss", "loss/train")
        val_loss = row_value(row, "val/loss", "validation/loss", "val_loss")
        best_epoch = row_value(row, "val/best_epoch", "best_epoch", "test_at_best_val/best_val_epoch")
        primary_metric = row_value(row, "primary_metric")
        primary_value = row_value(row, "val/primary_metric_value")

        for key, value in row.items():
            match = METRIC_RE.match(str(key))
            if not match:
                continue
            key_top_k = int(match.group("top_k"))
            retrieval_size = int(match.group("size"))
            split = match.group("split") or "unknown"
            variant = "balanced" if match.group("balanced") else "raw"
            if top_k and key_top_k != top_k:
                continue
            if retrieval_sizes is not None and retrieval_size not in retrieval_sizes:
                continue
            if splits is not None and split not in splits:
                continue
            if variant not in metric_variants:
                continue

            accuracy = to_float(value)
            if not math.isfinite(accuracy):
                continue

            prefix = "" if split == "unknown" else f"{split}/"
            n_samples = row_value(row, f"{prefix}n_samples_retrieval{retrieval_size}")
            n_skipped = row_value(row, f"{prefix}n_skipped_retrieval{retrieval_size}")
            records.append(
                {
                    "source": info.source,
                    "run_id": info.run_id,
                    "model": info.model,
                    "stage": info.stage,
                    "series": series_name(info.model, info.stage),
                    "run_dir": str(info.run_dir),
                    "history_file": str(info.history_file),
                    "source_row": source_row_index,
                    "epoch": to_intish(epoch),
                    "global_step": to_intish(global_step),
                    "split": split,
                    "metric_variant": variant,
                    "top_k": key_top_k,
                    "retrieval_size": retrieval_size,
                    "accuracy": accuracy,
                    "n_samples": to_intish(n_samples),
                    "n_skipped": to_intish(n_skipped),
                    "train_loss": to_float(train_loss),
                    "val_loss": to_float(val_loss),
                    "best_epoch": to_intish(best_epoch),
                    "primary_metric": "" if primary_metric is None else str(primary_metric),
                    "primary_metric_value": to_float(primary_value),
                }
            )
    return records


def normalise_pretraining_metric(key: str) -> tuple[str, str, str] | None:
    if "/" not in key:
        return None
    split, name = key.split("/", maxsplit=1)
    if split not in {"train", "val", "validation"}:
        return None
    if any(name.endswith(suffix) for suffix in PRETRAIN_SKIP_SUFFIXES):
        return None
    split = "val" if split == "validation" else split

    if name.endswith("_epoch"):
        name = name[: -len("_epoch")]
    if name in {"loss", "accuracy", "mask_ratio"}:
        return split, "global", name
    if re.fullmatch(r"accuracy_q\d+", name):
        return split, "global", name
    if name.endswith("_loss"):
        dataset = name[: -len("_loss")]
        if dataset:
            return split, dataset, "loss"
    if name.endswith("_acc"):
        dataset = name[: -len("_acc")]
        if dataset:
            return split, dataset, "accuracy"
    return None


def extract_pretraining_records(
    info: RunInfo,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if info.source.startswith("ds004408"):
        return []

    records: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(rows, start=1):
        epoch = row_value(row, "epoch", "trainer/epoch")
        global_step = row_value(row, "global_step", "step", "trainer/global_step")
        callback_stage = row_value(row, "stage")
        for key, value in row.items():
            parsed = normalise_pretraining_metric(str(key))
            if parsed is None:
                continue
            metric_split, dataset, metric = parsed
            metric_value = to_float(value)
            if not math.isfinite(metric_value):
                continue
            records.append(
                {
                    "source": info.source,
                    "run_id": info.run_id,
                    "model": info.model,
                    "stage": info.stage,
                    "series": series_name(info.model, info.stage),
                    "run_dir": str(info.run_dir),
                    "history_file": str(info.history_file),
                    "source_row": source_row_index,
                    "epoch": to_intish(epoch),
                    "global_step": to_intish(global_step),
                    "callback_stage": "" if callback_stage is None else str(callback_stage),
                    "split": metric_split,
                    "dataset": dataset,
                    "metric": metric,
                    "value": metric_value,
                }
            )
    return records


def series_name(model: str, stage: str) -> str:
    return f"{model} ({stage})" if stage else model


def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def sorted_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            str(row["source"]),
            str(row["run_id"]),
            str(row["split"]),
            str(row["metric_variant"]),
            int(row["retrieval_size"]),
            str(row["series"]),
            to_float(row["epoch"]),
            to_float(row["global_step"]),
            int(row["source_row"]),
        ),
    )


def group_for_comparison(records: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            row["source"],
            row["run_id"],
            row["split"],
            row["metric_variant"],
            row["top_k"],
            row["retrieval_size"],
        )
        grouped[key].append(row)
    return grouped


def group_for_run(records: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            row["source"],
            row["run_id"],
            row["model"],
            row["stage"],
            row["split"],
            row["metric_variant"],
            row["top_k"],
        )
        grouped[key].append(row)
    return grouped


def write_pivot_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
    line_field: str,
    *,
    value_field: str = "accuracy",
) -> None:
    epochs = sorted({str(row["epoch"]) for row in records}, key=lambda item: to_float(item))
    lines = sorted({str(row[line_field]) for row in records})
    values: dict[tuple[str, str], float] = {}
    for row in records:
        values[(str(row["epoch"]), str(row[line_field]))] = float(row[value_field])
    rows = []
    for epoch in epochs:
        out = {"epoch": epoch}
        for line in lines:
            value = values.get((epoch, line), math.nan)
            out[line] = "" if not math.isfinite(value) else value
        rows.append(out)
    write_csv(path, rows, ["epoch", *lines])


def latest_rows_by_epoch(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    latest_rank: dict[str, tuple[float, int]] = {}
    for row in records:
        epoch = str(row["epoch"])
        global_step = to_float(row.get("global_step"))
        if not math.isfinite(global_step):
            global_step = -math.inf
        try:
            source_row = int(row.get("source_row", 0))
        except (TypeError, ValueError):
            source_row = 0
        rank = (global_step, source_row)
        if epoch not in latest_rank or rank >= latest_rank[epoch]:
            latest[epoch] = row
            latest_rank[epoch] = rank
    return [latest[epoch] for epoch in sorted(latest, key=lambda item: to_float(item))]


def maybe_import_matplotlib() -> Any | None:
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "scrabrain-matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "scrabrain-cache"))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: matplotlib unavailable; skipping figures ({exc})")
        return None


def save_figure(fig: Any, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=220, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    return [str(path) for path in outputs]


def plot_comparison(plt: Any, path_stem: Path, records: Sequence[dict[str, Any]]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_series[str(row["series"])].append(row)
    for label, rows in sorted(by_series.items()):
        rows = sorted(rows, key=lambda row: to_float(row["epoch"]))
        xs = [to_float(row["epoch"]) for row in rows]
        ys = [to_float(row["accuracy"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=label)
    first = records[0]
    ylabel = "Balanced top-{top_k} accuracy" if first["metric_variant"] == "balanced" else "Top-{top_k} accuracy"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel.format(top_k=first["top_k"]))
    ax.set_title(
        f"{first['source']} {first['run_id']} · {first['split']} · "
        f"retrieval {first['retrieval_size']}"
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    outputs = save_figure(fig, path_stem)
    plt.close(fig)
    return outputs


def plot_run(plt: Any, path_stem: Path, records: Sequence[dict[str, Any]]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_size[int(row["retrieval_size"])].append(row)
    for size, rows in sorted(by_size.items()):
        rows = sorted(rows, key=lambda row: to_float(row["epoch"]))
        xs = [to_float(row["epoch"]) for row in rows]
        ys = [to_float(row["accuracy"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=f"retrieval {size}")
    first = records[0]
    ylabel = "Balanced top-{top_k} accuracy" if first["metric_variant"] == "balanced" else "Top-{top_k} accuracy"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel.format(top_k=first["top_k"]))
    title_stage = f" · {first['stage']}" if first["stage"] else ""
    ax.set_title(f"{first['source']} {first['run_id']} · {first['model']}{title_stage} · {first['split']}")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    outputs = save_figure(fig, path_stem)
    plt.close(fig)
    return outputs


def sorted_pretraining_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            str(row["source"]),
            str(row["run_id"]),
            str(row["stage"]),
            str(row["metric"]),
            str(row["split"]),
            str(row["dataset"]),
            str(row["series"]),
            to_float(row["epoch"]),
            to_float(row["global_step"]),
            int(row["source_row"]),
        ),
    )


def group_pretraining_global(records: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["dataset"] != "global":
            continue
        key = (row["source"], row["run_id"], row["stage"], row["split"], row["metric"])
        grouped[key].append(row)
    return grouped


def group_pretraining_by_dataset(records: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["dataset"] == "global":
            continue
        key = (
            row["source"],
            row["run_id"],
            row["model"],
            row["stage"],
            row["split"],
            row["metric"],
        )
        grouped[key].append(row)
    return grouped


def plot_pretraining_lines(
    plt: Any,
    path_stem: Path,
    records: Sequence[dict[str, Any]],
    *,
    line_field: str,
    title: str,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    by_line: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_line[str(row[line_field])].append(row)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    linestyles = ["-", "--", "-.", ":"]
    for label, rows in sorted(by_line.items()):
        rows = latest_rows_by_epoch(rows)
        xs = [to_float(row["epoch"]) for row in rows]
        ys = [to_float(row["value"]) for row in rows]
        index = len(ax.lines)
        ax.plot(
            xs,
            ys,
            marker="o",
            markevery=max(1, len(xs) // 10),
            linewidth=2.1,
            markersize=3.4,
            color=palette[index % len(palette)] if palette else None,
            linestyle=linestyles[index % len(linestyles)],
            label=label,
        )
    first = records[0]
    ax.set_xlabel("Epoch")
    ax.set_ylabel(str(first["metric"]))
    ax.set_title(title, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    outputs = save_figure(fig, path_stem)
    plt.close(fig)
    return outputs


def generate_group_outputs(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    make_figures: bool,
) -> tuple[list[str], list[str]]:
    csv_outputs: list[str] = []
    figure_outputs: list[str] = []
    plt = maybe_import_matplotlib() if make_figures else None

    for key, rows in sorted(group_for_comparison(records).items()):
        source, run_id, split, variant, top_k, retrieval_size = key
        stem = (
            output_dir
            / "comparison"
            / safe_slug(str(source))
            / safe_slug(str(run_id))
            / f"{safe_slug(str(split))}_{variant}_top{top_k}_retrieval{retrieval_size}"
        )
        write_pivot_csv(stem.with_suffix(".csv"), rows, "series")
        csv_outputs.append(str(stem.with_suffix(".csv")))
        if plt is not None:
            figure_outputs.extend(plot_comparison(plt, stem, rows))

    for key, rows in sorted(group_for_run(records).items()):
        source, run_id, model, stage, split, variant, top_k = key
        stem = (
            output_dir
            / "per_run"
            / safe_slug(str(source))
            / safe_slug(str(run_id))
            / f"{safe_slug(str(model))}_{safe_slug(str(stage))}_{safe_slug(str(split))}_{variant}_top{top_k}"
        )
        write_pivot_csv(stem.with_suffix(".csv"), rows, "retrieval_size")
        csv_outputs.append(str(stem.with_suffix(".csv")))
        if plt is not None:
            figure_outputs.extend(plot_run(plt, stem, rows))

    return csv_outputs, figure_outputs


def generate_pretraining_outputs(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    make_figures: bool,
) -> tuple[list[str], list[str]]:
    csv_outputs: list[str] = []
    figure_outputs: list[str] = []
    plt = maybe_import_matplotlib() if make_figures else None

    for key, rows in sorted(group_pretraining_global(records).items()):
        source, run_id, stage, split, metric = key
        stem = (
            output_dir
            / "pretraining"
            / "global"
            / safe_slug(str(source))
            / safe_slug(str(run_id))
            / f"{safe_slug(str(stage))}_{safe_slug(str(split))}_{safe_slug(str(metric))}"
        )
        write_pivot_csv(stem.with_suffix(".csv"), rows, "series", value_field="value")
        csv_outputs.append(str(stem.with_suffix(".csv")))
        if plt is not None:
            figure_outputs.extend(
                plot_pretraining_lines(
                    plt,
                    stem,
                    rows,
                    line_field="series",
                    title=f"{source} {run_id} · {stage} · {split}/{metric}",
                )
            )

    for key, rows in sorted(group_pretraining_by_dataset(records).items()):
        source, run_id, model, stage, split, metric = key
        stem = (
            output_dir
            / "pretraining"
            / "by_dataset"
            / safe_slug(str(source))
            / safe_slug(str(run_id))
            / f"{safe_slug(str(model))}_{safe_slug(str(stage))}_{safe_slug(str(split))}_{safe_slug(str(metric))}"
        )
        write_pivot_csv(stem.with_suffix(".csv"), rows, "dataset", value_field="value")
        csv_outputs.append(str(stem.with_suffix(".csv")))
        if plt is not None:
            figure_outputs.extend(
                plot_pretraining_lines(
                    plt,
                    stem,
                    rows,
                    line_field="dataset",
                    title=f"{source} {run_id} · {model} {stage} · {split}/{metric}",
                )
            )

    return csv_outputs, figure_outputs


def summarize_histories(
    history_files: Sequence[Path],
    records_by_file: dict[str, list[dict[str, Any]]],
    pretraining_records_by_file: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summary = []
    for history_file in history_files:
        info = infer_run_info(history_file)
        records = records_by_file.get(str(history_file), [])
        pretraining_records = pretraining_records_by_file.get(str(history_file), [])
        if records:
            status = "ok"
        elif pretraining_records:
            status = "ok_pretraining_only"
        else:
            status = "no_requested_metrics"
        summary.append(
            {
                "source": info.source,
                "run_id": info.run_id,
                "model": info.model,
                "stage": info.stage,
                "run_dir": str(info.run_dir),
                "history_file": str(history_file),
                "top10_records": len(records),
                "pretraining_records": len(pretraining_records),
                "status": status,
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    scan_roots = args.scan_root or default_scan_roots()
    retrieval_sizes = set(args.retrieval_sizes) if args.retrieval_sizes else None
    splits = set(args.splits) if args.splits else None
    metric_variants = set(args.metric_variants)

    history_files = discover_history_files(scan_roots, args.run_dir, args.include_pattern)
    all_records: list[dict[str, Any]] = []
    all_pretraining_records: list[dict[str, Any]] = []
    records_by_file: dict[str, list[dict[str, Any]]] = {}
    pretraining_records_by_file: dict[str, list[dict[str, Any]]] = {}

    for history_file in history_files:
        info = infer_run_info(history_file)
        rows = load_history(history_file)
        records = extract_records(
            info,
            rows,
            top_k=args.top_k,
            retrieval_sizes=retrieval_sizes,
            splits=splits,
            metric_variants=metric_variants,
        )
        records_by_file[str(history_file)] = records
        all_records.extend(records)
        pretraining_records = extract_pretraining_records(info, rows)
        pretraining_records_by_file[str(history_file)] = pretraining_records
        all_pretraining_records.extend(pretraining_records)

    all_records = sorted_records(all_records)
    all_pretraining_records = sorted_pretraining_records(all_pretraining_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_csv = args.output_dir / "top10_training_curves_long.csv"
    fieldnames = [
        "source",
        "run_id",
        "model",
        "stage",
        "series",
        "epoch",
        "global_step",
        "split",
        "metric_variant",
        "top_k",
        "retrieval_size",
        "accuracy",
        "n_samples",
        "n_skipped",
        "train_loss",
        "val_loss",
        "best_epoch",
        "primary_metric",
        "primary_metric_value",
        "run_dir",
        "history_file",
        "source_row",
    ]
    write_csv(long_csv, all_records, fieldnames)

    pretraining_csv = args.output_dir / "pretraining_metrics_long.csv"
    pretraining_fieldnames = [
        "source",
        "run_id",
        "model",
        "stage",
        "series",
        "epoch",
        "global_step",
        "callback_stage",
        "split",
        "dataset",
        "metric",
        "value",
        "run_dir",
        "history_file",
        "source_row",
    ]
    write_csv(pretraining_csv, all_pretraining_records, pretraining_fieldnames)

    summary_rows = summarize_histories(
        history_files,
        records_by_file,
        pretraining_records_by_file,
    )
    summary_csv = args.output_dir / "top10_training_curves_manifest.csv"
    write_csv(
        summary_csv,
        summary_rows,
        [
            "source",
            "run_id",
            "model",
            "stage",
            "run_dir",
            "history_file",
            "top10_records",
            "pretraining_records",
            "status",
        ],
    )

    pivot_csvs: list[str] = []
    figures: list[str] = []
    if all_records:
        pivot_csvs, figures = generate_group_outputs(
            all_records,
            args.output_dir,
            make_figures=not args.no_figures,
        )
    pretraining_csvs: list[str] = []
    pretraining_figures: list[str] = []
    if all_pretraining_records:
        pretraining_csvs, pretraining_figures = generate_pretraining_outputs(
            all_pretraining_records,
            args.output_dir,
            make_figures=not args.no_figures,
        )

    manifest = {
        "history_files_scanned": [str(path) for path in history_files],
        "n_history_files": len(history_files),
        "n_top10_records": len(all_records),
        "n_pretraining_records": len(all_pretraining_records),
        "long_csv": str(long_csv),
        "pretraining_csv": str(pretraining_csv),
        "manifest_csv": str(summary_csv),
        "pivot_csvs": pivot_csvs,
        "pretraining_pivot_csvs": pretraining_csvs,
        "figures": figures,
        "pretraining_figures": pretraining_figures,
        "filters": {
            "include_pattern": args.include_pattern,
            "top_k": "all" if args.top_k == 0 else args.top_k,
            "retrieval_sizes": "all" if retrieval_sizes is None else sorted(retrieval_sizes),
            "splits": "all" if splits is None else sorted(splits),
            "metric_variants": sorted(metric_variants),
        },
    }
    manifest_json = args.output_dir / "top10_training_curves_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Scanned histories: {len(history_files)}")
    print(f"Exported top-10 rows: {len(all_records)}")
    print(f"Exported pretraining rows: {len(all_pretraining_records)}")
    print(f"Long CSV: {long_csv}")
    print(f"Pretraining CSV: {pretraining_csv}")
    print(f"Manifest: {manifest_json}")
    if pivot_csvs:
        print(f"Pivot CSVs: {len(pivot_csvs)}")
    if figures:
        print(f"Figures: {len(figures)}")
    if pretraining_csvs:
        print(f"Pretraining pivot CSVs: {len(pretraining_csvs)}")
    if pretraining_figures:
        print(f"Pretraining figures: {len(pretraining_figures)}")
    if not all_records and not all_pretraining_records:
        print("No requested top-10 or pretraining metric columns were found. Check manifest CSV for scanned files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
