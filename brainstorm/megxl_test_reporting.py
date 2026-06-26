"""MEG-XL-compatible reporting for downstream word-decoding experiments.

The MEG-XL paper evaluates word decoding with macro-averaged top-10 retrieval
accuracy over the 50 most frequent words. The appendix repeats the experiment
with a 250-word retrieval vocabulary. Across repeated seeds, figures report the
mean and standard error of the mean (SEM).

This module enriches the evaluator output with the corresponding chance and
chance-normalised quantities, writes paper-oriented tables, and automatically
creates:

* per-run validation curves and final test plots;
* Figure-3-style top-50 data-efficiency comparisons;
* Figure-6-style top-250 data-efficiency comparisons;
* mean/std/SEM summaries and pairwise Welch tests when repeated seeds exist.

It deliberately does not attempt to reproduce the paper's linear-probe,
masked-token-prediction, or attention-analysis figures because those require
separate pre-training/context ablations and attention tensors, not the final
word-decoding test outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


DEFAULT_RETRIEVAL_SIZES = (50, 250)
DEFAULT_TOP_K = 10

PAPER_MODEL_LABELS = {
    "random_init": "Arquitectura aleatoria",
    "eeg_from_scratch": "Checkpoint EEG desde cero",
    "eeg_pretrained": "Checkpoint EEG preentrenado",
}

PAPER_MODEL_COLORS = {
    "random_init": "#6f6f6f",
    "eeg_from_scratch": "#4477aa",
    "eeg_pretrained": "#cc3355",
}


@dataclass(frozen=True)
class RunRecord:
    """Paper-relevant final result from one completed fine-tuning run."""

    label: str
    display_name: str
    run_dir: Path
    seed: int
    train_pct: float
    retrieval_size: int
    top_k: int
    balanced_accuracy: float
    accuracy: float
    chance_accuracy: float
    balanced_x_chance: float
    accuracy_x_chance: float
    n_samples: int
    n_skipped: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def chance_top_k_accuracy(retrieval_size: int, top_k: int = DEFAULT_TOP_K) -> float:
    """Random-retrieval top-k accuracy for a fixed retrieval vocabulary."""

    if retrieval_size <= 0:
        raise ValueError(f"retrieval_size must be positive, got {retrieval_size}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return min(top_k, retrieval_size) / retrieval_size


def enrich_test_metrics(
    metrics: Mapping[str, Any],
    *,
    retrieval_sizes: Sequence[int] = DEFAULT_RETRIEVAL_SIZES,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Add paper-oriented chance, margin, and uplift metrics.

    The balanced top-k retrieval accuracy itself is produced by the core
    evaluator. This function adds quantities needed to interpret it against the
    random baseline and to produce the paper-style plots.
    """

    enriched = dict(metrics)
    for retrieval_size in retrieval_sizes:
        chance = chance_top_k_accuracy(retrieval_size, top_k)
        accuracy_key = f"top{top_k}_accuracy_retrieval{retrieval_size}"
        balanced_key = f"balanced_top{top_k}_accuracy_retrieval{retrieval_size}"

        accuracy = _to_float(enriched.get(accuracy_key))
        balanced = _to_float(enriched.get(balanced_key))

        enriched[f"chance_top{top_k}_accuracy_retrieval{retrieval_size}"] = chance
        if math.isfinite(accuracy):
            enriched[f"top{top_k}_accuracy_above_chance_retrieval{retrieval_size}"] = accuracy - chance
            enriched[f"top{top_k}_accuracy_x_chance_retrieval{retrieval_size}"] = accuracy / chance
        if math.isfinite(balanced):
            enriched[
                f"balanced_top{top_k}_accuracy_above_chance_retrieval{retrieval_size}"
            ] = balanced - chance
            enriched[
                f"balanced_top{top_k}_accuracy_x_chance_retrieval{retrieval_size}"
            ] = balanced / chance

    return enriched


def _resolved_config(run_dir: Path) -> Any | None:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return None
    try:
        return OmegaConf.load(config_path)
    except Exception:
        return None


def _run_train_pct(run_dir: Path, final_results: Mapping[str, Any]) -> float:
    cfg = _resolved_config(run_dir)
    if cfg is not None:
        try:
            return float(cfg.data.train_pct)
        except (AttributeError, TypeError, ValueError):
            pass
    return _to_float(final_results.get("train_pct"), 1.0)


def _run_seed(run_dir: Path, final_results: Mapping[str, Any]) -> int:
    if "seed" in final_results:
        return _to_int(final_results.get("seed"), 0)
    cfg = _resolved_config(run_dir)
    if cfg is not None:
        try:
            return int(cfg.seed)
        except (AttributeError, TypeError, ValueError):
            pass
    return 0


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No results.\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered = "" if not math.isfinite(value) else f"{value:.6f}"
            else:
                rendered = str(value)
            values.append(rendered.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_epoch_history(run_dir: Path) -> list[dict[str, str]]:
    for filename in ("epoch_metrics.csv", "metrics_history.csv"):
        path = run_dir / filename
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
    return []


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def _plot_run_training_curves(
    run_dir: Path,
    *,
    retrieval_sizes: Sequence[int],
    top_k: int,
    final_metrics: Mapping[str, Any],
) -> list[str]:
    history = _load_epoch_history(run_dir)
    if not history:
        return []

    artifacts: list[str] = []
    epochs = np.asarray([_to_float(row.get("epoch")) for row in history], dtype=float)

    for retrieval_size in retrieval_sizes:
        val_key = f"val/balanced_top{top_k}_accuracy_retrieval{retrieval_size}"
        val_values = np.asarray([_to_float(row.get(val_key)) for row in history], dtype=float)
        valid = np.isfinite(epochs) & np.isfinite(val_values)
        if not valid.any():
            continue

        chance = chance_top_k_accuracy(retrieval_size, top_k)
        final_test = _to_float(
            final_metrics.get(f"balanced_top{top_k}_accuracy_retrieval{retrieval_size}")
        )

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(epochs[valid], val_values[valid], marker="o", label="Validación")
        best_idx = int(np.nanargmax(np.where(valid, val_values, np.nan)))
        ax.scatter(
            [epochs[best_idx]],
            [val_values[best_idx]],
            marker="*",
            s=130,
            zorder=4,
            label="Mejor validación",
        )
        ax.axhline(chance, linestyle="--", linewidth=1.2, label=f"Azar ({chance:.0%})")
        if math.isfinite(final_test):
            ax.axhline(
                final_test,
                linestyle=":",
                linewidth=1.7,
                label=f"Test final ({final_test:.1%})",
            )
        ax.set_xlabel("Época")
        ax.set_ylabel(f"Balanced top-{top_k} accuracy")
        ax.set_title(f"Evolución de validación · vocabulario {retrieval_size}")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        artifacts.extend(
            _save_figure(
                fig,
                run_dir / f"training_balanced_top{top_k}_retrieval{retrieval_size}",
            )
        )

    train_losses = np.asarray(
        [_to_float(row.get("train/loss")) for row in history], dtype=float
    )
    val_losses = np.asarray([_to_float(row.get("val/loss")) for row in history], dtype=float)
    train_valid = np.isfinite(epochs) & np.isfinite(train_losses)
    val_valid = np.isfinite(epochs) & np.isfinite(val_losses)
    if train_valid.any() or val_valid.any():
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if train_valid.any():
            ax.plot(epochs[train_valid], train_losses[train_valid], label="Entrenamiento")
        if val_valid.any():
            ax.plot(epochs[val_valid], val_losses[val_valid], label="Validación")
        ax.set_xlabel("Época")
        ax.set_ylabel("D-SigLIP loss")
        ax.set_title("Pérdida durante el fine-tuning")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        artifacts.extend(_save_figure(fig, run_dir / "training_loss"))

    return artifacts


def _plot_run_final_test(
    run_dir: Path,
    *,
    retrieval_sizes: Sequence[int],
    top_k: int,
    metrics: Mapping[str, Any],
) -> list[str]:
    sizes = list(retrieval_sizes)
    balanced = np.asarray(
        [
            _to_float(metrics.get(f"balanced_top{top_k}_accuracy_retrieval{size}"))
            for size in sizes
        ],
        dtype=float,
    )
    micro = np.asarray(
        [_to_float(metrics.get(f"top{top_k}_accuracy_retrieval{size}")) for size in sizes],
        dtype=float,
    )
    chance = np.asarray([chance_top_k_accuracy(size, top_k) for size in sizes], dtype=float)

    x = np.arange(len(sizes), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar(x - width / 2, balanced, width, label="Balanced (paper)")
    ax.bar(x + width / 2, micro, width, label="Micro")
    ax.scatter(x, chance, marker="x", s=70, linewidths=2, label="Azar", zorder=4)
    ax.set_xticks(x, [f"Top {size} palabras" for size in sizes])
    ax.set_ylabel(f"Top-{top_k} accuracy")
    ax.set_title("Evaluación final del mejor checkpoint")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for positions, values in ((x - width / 2, balanced), (x + width / 2, micro)):
        for xpos, value in zip(positions, values):
            if math.isfinite(value):
                ax.text(xpos, value, f"{value:.1%}", ha="center", va="bottom", fontsize=8)

    return _save_figure(fig, run_dir / f"final_test_top{top_k}_accuracy")


def generate_run_report(
    run_dir: str | Path,
    *,
    retrieval_sizes: Sequence[int] = DEFAULT_RETRIEVAL_SIZES,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Enrich one completed run and generate all per-run report artifacts."""

    run_dir = Path(run_dir)
    final_path = run_dir / "final_results.json"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing final result file: {final_path}")

    final_results = _read_json(final_path)
    test_metrics = final_results.get("test_metrics", {})
    if not isinstance(test_metrics, dict):
        raise ValueError(f"test_metrics is not an object in {final_path}")

    enriched_metrics = enrich_test_metrics(
        test_metrics,
        retrieval_sizes=retrieval_sizes,
        top_k=top_k,
    )
    train_pct = _run_train_pct(run_dir, final_results)
    seed = _run_seed(run_dir, final_results)

    final_results["test_metrics"] = enriched_metrics
    final_results["train_pct"] = train_pct
    final_results["paper_evaluation"] = {
        "primary_metric": f"balanced_top{top_k}_accuracy_retrieval50",
        "retrieval_set_sizes": [int(size) for size in retrieval_sizes],
        "top_k": int(top_k),
        "chance_definition": "min(top_k, retrieval_set_size) / retrieval_set_size",
        "aggregation_across_classes": "macro average over observed target word classes",
        "checkpoint_selection": "best validation checkpoint",
        "seed": seed,
        "train_pct": train_pct,
        "sem_note": "SEM is computed across repeated seeds only in the comparison report.",
    }
    _write_json(final_path, final_results)

    metric_rows: list[dict[str, Any]] = []
    for retrieval_size in retrieval_sizes:
        metric_rows.append(
            {
                "retrieval_size": int(retrieval_size),
                "top_k": int(top_k),
                "balanced_top_k_accuracy": _to_float(
                    enriched_metrics.get(
                        f"balanced_top{top_k}_accuracy_retrieval{retrieval_size}"
                    )
                ),
                "top_k_accuracy": _to_float(
                    enriched_metrics.get(f"top{top_k}_accuracy_retrieval{retrieval_size}")
                ),
                "chance_accuracy": chance_top_k_accuracy(retrieval_size, top_k),
                "balanced_accuracy_above_chance": _to_float(
                    enriched_metrics.get(
                        f"balanced_top{top_k}_accuracy_above_chance_retrieval{retrieval_size}"
                    )
                ),
                "balanced_accuracy_x_chance": _to_float(
                    enriched_metrics.get(
                        f"balanced_top{top_k}_accuracy_x_chance_retrieval{retrieval_size}"
                    )
                ),
                "accuracy_above_chance": _to_float(
                    enriched_metrics.get(
                        f"top{top_k}_accuracy_above_chance_retrieval{retrieval_size}"
                    )
                ),
                "accuracy_x_chance": _to_float(
                    enriched_metrics.get(
                        f"top{top_k}_accuracy_x_chance_retrieval{retrieval_size}"
                    )
                ),
                "n_samples": _to_int(
                    enriched_metrics.get(f"n_samples_retrieval{retrieval_size}")
                ),
                "n_skipped": _to_int(
                    enriched_metrics.get(f"n_skipped_retrieval{retrieval_size}")
                ),
            }
        )

    _write_rows_csv(run_dir / "paper_test_metrics.csv", metric_rows)
    _write_rows_markdown(run_dir / "paper_test_metrics.md", metric_rows)
    _write_json(
        run_dir / "paper_test_metrics.json",
        {
            "experiment_name": final_results.get("experiment_name", run_dir.name),
            "seed": seed,
            "train_pct": train_pct,
            "metrics": metric_rows,
        },
    )

    figure_paths = []
    figure_paths.extend(
        _plot_run_training_curves(
            run_dir,
            retrieval_sizes=retrieval_sizes,
            top_k=top_k,
            final_metrics=enriched_metrics,
        )
    )
    figure_paths.extend(
        _plot_run_final_test(
            run_dir,
            retrieval_sizes=retrieval_sizes,
            top_k=top_k,
            metrics=enriched_metrics,
        )
    )

    manifest = {
        "run_dir": str(run_dir),
        "final_results": str(final_path),
        "paper_metrics_csv": str(run_dir / "paper_test_metrics.csv"),
        "paper_metrics_markdown": str(run_dir / "paper_test_metrics.md"),
        "figures": figure_paths,
    }
    _write_json(run_dir / "paper_report_manifest.json", manifest)
    return manifest


def _display_name(label: str) -> str:
    return PAPER_MODEL_LABELS.get(label, label.replace("_", " ").strip().title())


def _load_run_records(
    run_specs: Sequence[tuple[str, Path]],
    *,
    retrieval_sizes: Sequence[int],
    top_k: int,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for label, run_dir in run_specs:
        final_path = run_dir / "final_results.json"
        final_results = _read_json(final_path)
        metrics = final_results.get("test_metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"test_metrics is not an object in {final_path}")
        metrics = enrich_test_metrics(metrics, retrieval_sizes=retrieval_sizes, top_k=top_k)
        train_pct = _run_train_pct(run_dir, final_results)
        seed = _run_seed(run_dir, final_results)

        for retrieval_size in retrieval_sizes:
            balanced = _to_float(
                metrics.get(f"balanced_top{top_k}_accuracy_retrieval{retrieval_size}")
            )
            accuracy = _to_float(
                metrics.get(f"top{top_k}_accuracy_retrieval{retrieval_size}")
            )
            chance = chance_top_k_accuracy(retrieval_size, top_k)
            records.append(
                RunRecord(
                    label=label,
                    display_name=_display_name(label),
                    run_dir=run_dir,
                    seed=seed,
                    train_pct=train_pct,
                    retrieval_size=int(retrieval_size),
                    top_k=int(top_k),
                    balanced_accuracy=balanced,
                    accuracy=accuracy,
                    chance_accuracy=chance,
                    balanced_x_chance=balanced / chance if math.isfinite(balanced) else math.nan,
                    accuracy_x_chance=accuracy / chance if math.isfinite(accuracy) else math.nan,
                    n_samples=_to_int(metrics.get(f"n_samples_retrieval{retrieval_size}")),
                    n_skipped=_to_int(metrics.get(f"n_skipped_retrieval{retrieval_size}")),
                )
            )
    return records


def _record_rows(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    return [
        {
            "model": record.label,
            "model_display": record.display_name,
            "run_dir": str(record.run_dir),
            "seed": record.seed,
            "train_pct": record.train_pct,
            "retrieval_size": record.retrieval_size,
            "top_k": record.top_k,
            "balanced_top_k_accuracy": record.balanced_accuracy,
            "top_k_accuracy": record.accuracy,
            "chance_accuracy": record.chance_accuracy,
            "balanced_accuracy_x_chance": record.balanced_x_chance,
            "accuracy_x_chance": record.accuracy_x_chance,
            "n_samples": record.n_samples,
            "n_skipped": record.n_skipped,
        }
        for record in records
    ]


def _summary_rows(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, int, int], list[RunRecord]] = {}
    for record in records:
        key = (
            record.label,
            record.display_name,
            record.train_pct,
            record.retrieval_size,
            record.top_k,
        )
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: (item[0][3], item[0][2], item[0][0])):
        label, display_name, train_pct, retrieval_size, top_k = key
        balanced_values = np.asarray(
            _finite(record.balanced_accuracy for record in group), dtype=float
        )
        accuracy_values = np.asarray(_finite(record.accuracy for record in group), dtype=float)
        n = int(len(balanced_values))
        balanced_std = float(np.std(balanced_values, ddof=1)) if n > 1 else math.nan
        balanced_sem = balanced_std / math.sqrt(n) if n > 1 else math.nan
        accuracy_std = (
            float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else math.nan
        )
        accuracy_sem = (
            accuracy_std / math.sqrt(len(accuracy_values))
            if len(accuracy_values) > 1
            else math.nan
        )
        mean_balanced = float(np.mean(balanced_values)) if n else math.nan
        mean_accuracy = float(np.mean(accuracy_values)) if len(accuracy_values) else math.nan
        chance = chance_top_k_accuracy(retrieval_size, top_k)
        rows.append(
            {
                "model": label,
                "model_display": display_name,
                "train_pct": train_pct,
                "retrieval_size": retrieval_size,
                "top_k": top_k,
                "n_seeds": n,
                "balanced_mean": mean_balanced,
                "balanced_std": balanced_std,
                "balanced_sem": balanced_sem,
                "balanced_ci95_low": mean_balanced - 1.96 * balanced_sem
                if math.isfinite(balanced_sem)
                else math.nan,
                "balanced_ci95_high": mean_balanced + 1.96 * balanced_sem
                if math.isfinite(balanced_sem)
                else math.nan,
                "accuracy_mean": mean_accuracy,
                "accuracy_std": accuracy_std,
                "accuracy_sem": accuracy_sem,
                "chance_accuracy": chance,
                "balanced_mean_x_chance": mean_balanced / chance
                if math.isfinite(mean_balanced)
                else math.nan,
            }
        )
    return rows


def _welch_rows(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    try:
        from scipy.stats import ttest_ind
    except Exception:
        return []

    grouped: dict[tuple[float, int, int, str], list[float]] = {}
    display_names: dict[str, str] = {}
    for record in records:
        if not math.isfinite(record.balanced_accuracy):
            continue
        key = (record.train_pct, record.retrieval_size, record.top_k, record.label)
        grouped.setdefault(key, []).append(record.balanced_accuracy)
        display_names[record.label] = record.display_name

    rows: list[dict[str, Any]] = []
    settings = sorted({key[:3] for key in grouped})
    for train_pct, retrieval_size, top_k in settings:
        labels = sorted(
            key[3]
            for key in grouped
            if key[:3] == (train_pct, retrieval_size, top_k)
        )
        for left, right in combinations(labels, 2):
            left_values = grouped[(train_pct, retrieval_size, top_k, left)]
            right_values = grouped[(train_pct, retrieval_size, top_k, right)]
            if len(left_values) >= 2 and len(right_values) >= 2:
                result = ttest_ind(left_values, right_values, equal_var=False)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            else:
                statistic = math.nan
                p_value = math.nan
            rows.append(
                {
                    "train_pct": train_pct,
                    "retrieval_size": retrieval_size,
                    "top_k": top_k,
                    "model_a": left,
                    "model_a_display": display_names[left],
                    "n_a": len(left_values),
                    "mean_a": float(np.mean(left_values)),
                    "model_b": right,
                    "model_b_display": display_names[right],
                    "n_b": len(right_values),
                    "mean_b": float(np.mean(right_values)),
                    "welch_t": statistic,
                    "p_value": p_value,
                    "significant_p_lt_0_05": bool(p_value < 0.05)
                    if math.isfinite(p_value)
                    else "",
                    "note": "Requires at least two seeds per model"
                    if not math.isfinite(p_value)
                    else "",
                }
            )
    return rows


def _plot_paper_data_efficiency(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    retrieval_size: int,
    top_k: int,
    output_dir: Path,
    label_order: Sequence[str],
) -> list[str]:
    relevant = [
        row
        for row in summary_rows
        if _to_int(row.get("retrieval_size")) == retrieval_size
        and _to_int(row.get("top_k")) == top_k
    ]
    if not relevant:
        return []

    train_pcts = sorted({_to_float(row.get("train_pct")) for row in relevant})
    multiple_ratios = len(train_pcts) > 1
    chance = chance_top_k_accuracy(retrieval_size, top_k)

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    if multiple_ratios:
        for label in label_order:
            model_rows = sorted(
                [row for row in relevant if row.get("model") == label],
                key=lambda row: _to_float(row.get("train_pct")),
            )
            if not model_rows:
                continue
            x = np.asarray([_to_float(row.get("train_pct")) for row in model_rows])
            y = np.asarray([_to_float(row.get("balanced_mean")) for row in model_rows])
            sem = np.asarray([_to_float(row.get("balanced_sem")) for row in model_rows])
            yerr = np.where(np.isfinite(sem), sem, 0.0)
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=PAPER_MODEL_COLORS.get(label),
                label=_display_name(label),
            )
        if all(value > 0 for value in train_pcts):
            ax.set_xscale("log")
        ax.set_xlabel("Proporción de datos de entrenamiento")
    else:
        model_rows = []
        for label in label_order:
            matches = [row for row in relevant if row.get("model") == label]
            if matches:
                model_rows.append(matches[0])
        x = np.arange(len(model_rows), dtype=float)
        y = np.asarray([_to_float(row.get("balanced_mean")) for row in model_rows])
        sem = np.asarray([_to_float(row.get("balanced_sem")) for row in model_rows])
        yerr = np.where(np.isfinite(sem), sem, 0.0)
        colors = [PAPER_MODEL_COLORS.get(str(row.get("model"))) for row in model_rows]
        ax.bar(x, y, yerr=yerr, capsize=4, color=colors)
        ax.set_xticks(
            x,
            [str(row.get("model_display")) for row in model_rows],
            rotation=12,
            ha="right",
        )
        ax.set_xlabel(f"Fine-tuning con {train_pcts[0]:.0%} de train")
        for xpos, value in zip(x, y):
            if math.isfinite(value):
                ax.text(xpos, value, f"{value:.1%}", ha="center", va="bottom", fontsize=9)

    ax.axhline(chance, linestyle="--", linewidth=1.2, label=f"Azar ({chance:.0%})")
    ax.set_ylabel(f"Balanced top-{top_k} accuracy")
    figure_number = "3" if retrieval_size == 50 else "6"
    ax.set_title(
        f"MEG-XL Figura {figure_number} · vocabulario de {retrieval_size} palabras"
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save_figure(
        fig,
        output_dir / f"megxl_figure{figure_number}_top{top_k}_retrieval{retrieval_size}",
    )


def generate_comparison_report(
    run_specs: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    retrieval_sizes: Sequence[int] = DEFAULT_RETRIEVAL_SIZES,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Aggregate completed runs and generate MEG-XL-style comparison artifacts."""

    normalised_specs = [(label, Path(path)) for label, path in run_specs]
    if not normalised_specs:
        raise ValueError("At least one run must be supplied")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_run_records(
        normalised_specs,
        retrieval_sizes=retrieval_sizes,
        top_k=top_k,
    )
    long_rows = _record_rows(records)
    summary_rows = _summary_rows(records)
    welch_rows = _welch_rows(records)

    _write_rows_csv(output_dir / "weissbart_three_way_test_metrics.csv", long_rows)
    _write_rows_markdown(output_dir / "weissbart_three_way_test_metrics.md", long_rows)
    _write_rows_csv(output_dir / "megxl_paper_metrics_summary.csv", summary_rows)
    _write_rows_markdown(output_dir / "megxl_paper_metrics_summary.md", summary_rows)
    _write_rows_csv(output_dir / "megxl_pairwise_welch_tests.csv", welch_rows)
    _write_rows_markdown(output_dir / "megxl_pairwise_welch_tests.md", welch_rows)

    label_order = []
    for label, _path in normalised_specs:
        if label not in label_order:
            label_order.append(label)

    figure_paths: list[str] = []
    for retrieval_size in retrieval_sizes:
        figure_paths.extend(
            _plot_paper_data_efficiency(
                summary_rows,
                retrieval_size=int(retrieval_size),
                top_k=top_k,
                output_dir=output_dir,
                label_order=label_order,
            )
        )

    manifest = {
        "runs": [{"label": label, "run_dir": str(path)} for label, path in normalised_specs],
        "top_k": top_k,
        "retrieval_sizes": [int(size) for size in retrieval_sizes],
        "long_metrics_csv": str(output_dir / "weissbart_three_way_test_metrics.csv"),
        "summary_csv": str(output_dir / "megxl_paper_metrics_summary.csv"),
        "welch_tests_csv": str(output_dir / "megxl_pairwise_welch_tests.csv"),
        "figures": figure_paths,
        "sem_definition": "sample standard deviation across seeds divided by sqrt(n_seeds)",
        "paper_scope_note": (
            "Figures 3 and 6 are generated from word-decoding results. "
            "Figures 4, 5, and 7 require separate context/pre-training or attention experiments."
        ),
    }
    _write_json(output_dir / "megxl_paper_report_manifest.json", manifest)
    return manifest


def parse_comparison_runs(spec: str) -> list[tuple[str, Path]]:
    """Parse ``label=path;label=path`` from an environment variable."""

    runs: list[tuple[str, Path]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid comparison run specification: {item!r}")
        label, path = item.split("=", 1)
        runs.append((label.strip(), Path(path.strip())))
    return runs


def maybe_generate_comparison_from_environment() -> dict[str, Any] | None:
    """Generate the combined report when requested by the orchestration script."""

    specification = os.environ.get("MEGXL_COMPARISON_RUNS", "").strip()
    if not specification:
        return None
    output_dir = os.environ.get("MEGXL_COMPARISON_OUTPUT", "").strip()
    if not output_dir:
        raise ValueError("MEGXL_COMPARISON_OUTPUT is required with MEGXL_COMPARISON_RUNS")
    return generate_comparison_report(parse_comparison_runs(specification), output_dir)


def _parse_run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must have the form label=/path/to/run")
    label, path = value.split("=", 1)
    return label, Path(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Report one completed run")
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    run_parser.add_argument(
        "--retrieval-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_RETRIEVAL_SIZES),
    )

    compare_parser = subparsers.add_parser("compare", help="Aggregate several runs")
    compare_parser.add_argument(
        "--run",
        dest="runs",
        type=_parse_run_argument,
        action="append",
        required=True,
        help="Repeat as label=/path/to/run",
    )
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    compare_parser.add_argument(
        "--retrieval-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_RETRIEVAL_SIZES),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run":
        manifest = generate_run_report(
            args.run_dir,
            retrieval_sizes=args.retrieval_sizes,
            top_k=args.top_k,
        )
    else:
        manifest = generate_comparison_report(
            args.runs,
            args.output_dir,
            retrieval_sizes=args.retrieval_sizes,
            top_k=args.top_k,
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
