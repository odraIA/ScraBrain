"""Alice EEG fine-tuning entry point with MEG-XL and paper-comparison reports."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

from omegaconf import DictConfig

from brainstorm import evaluate_criss_cross_word_classification as evaluator
from brainstorm import optimized_word_finetuning as optimized
from brainstorm.data.alice_eeg_word_aligned_dataset import AliceEEGWordAlignedDataset
from brainstorm.megxl_test_reporting import (
    generate_comparison_report,
    generate_run_report,
    parse_comparison_runs,
)


ARTICLE_RETRIEVAL_SIZE = 601
ARTICLE_TOP1 = 0.0410
ARTICLE_TOP10 = 0.2682

_BASE_GET_DATASET_CLASS = evaluator.get_dataset_class
_BASE_GET_DEFAULT_MAX_CHANNEL_DIM = evaluator.get_default_max_channel_dim
_BASE_GET_DATASET_EXTRA_KWARGS = evaluator.get_dataset_extra_kwargs
_BASE_GET_NUM_SENSOR_TYPES = evaluator.get_num_sensor_types_for_config
_BASE_OPTIMIZED_EVALUATE_EPOCH = optimized._evaluate_epoch


def get_dataset_class(dataset_type: str):
    if dataset_type == "alice_eeg":
        return AliceEEGWordAlignedDataset
    return _BASE_GET_DATASET_CLASS(dataset_type)


def get_default_max_channel_dim(dataset_type: str) -> int:
    if dataset_type == "alice_eeg":
        return 64
    return _BASE_GET_DEFAULT_MAX_CHANNEL_DIM(dataset_type)


def get_dataset_extra_kwargs(dataset_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if dataset_type == "alice_eeg":
        return {
            "eeg_sensor_type": cfg.data.get("eeg_sensor_type", "grad"),
            "dataset_name": cfg.data.get("dataset_name", "alice_eeg"),
            "task_mode": cfg.data.get("task_mode", "listening"),
            "tokenizer_name": cfg.model.get("tokenizer_name", "biocodec"),
            "subject_selection": cfg.data.get("subject_selection", "main"),
        }
    return _BASE_GET_DATASET_EXTRA_KWARGS(dataset_type, cfg)


def get_num_sensor_types_for_config(cfg: DictConfig) -> int:
    num_sensor_types = _BASE_GET_NUM_SENSOR_TYPES(cfg)
    if cfg.data.get("dataset_type") == "alice_eeg":
        sensor_type_id = evaluator.resolve_sensor_type_id(
            cfg.data.get("eeg_sensor_type", "grad")
        )
        num_sensor_types = max(num_sensor_types, sensor_type_id + 1)
    return num_sensor_types


def _alice_evaluate_epoch(
    criss_cross_model,
    word_mlp,
    dataloader,
    vocab_embeddings_device,
    vocab_embeddings_cpu,
    criterion,
    device,
    retrieval_set_sizes,
    k,
    downsample_ratio,
    mixed_precision,
    amp_dtype,
    features_only_forward,
    description,
):
    """Run normal Top-10 metrics and add the paper's 601-way Top-1 on final test."""

    metrics = _BASE_OPTIMIZED_EVALUATE_EPOCH(
        criss_cross_model,
        word_mlp,
        dataloader,
        vocab_embeddings_device,
        vocab_embeddings_cpu,
        criterion,
        device,
        retrieval_set_sizes,
        k,
        downsample_ratio,
        mixed_precision,
        amp_dtype,
        features_only_forward,
        description,
    )
    if description != "Final test":
        return metrics

    top1_metrics = _BASE_OPTIMIZED_EVALUATE_EPOCH(
        criss_cross_model,
        word_mlp,
        dataloader,
        vocab_embeddings_device,
        vocab_embeddings_cpu,
        criterion,
        device,
        [ARTICLE_RETRIEVAL_SIZE],
        1,
        downsample_ratio,
        mixed_precision,
        amp_dtype,
        features_only_forward,
        "Final test Top-1 / 601",
    )
    for key, value in top1_metrics.items():
        if "top1" in key or key.startswith("n_samples_retrieval601") or key.startswith(
            "n_skipped_retrieval601"
        ):
            metrics[key] = value
    return metrics


def _cli_override(name: str) -> str | None:
    prefix = f"{name}="
    for argument in reversed(sys.argv[1:]):
        if argument.startswith(prefix):
            return argument[len(prefix) :].strip().strip('"').strip("'")
    return None


def _read_json(path: Path) -> Dict[str, Any]:
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


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rows_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No results.\n", encoding="utf-8")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "" if not math.isfinite(value) else f"{value:.6f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_row(
    *,
    metric: str,
    our_value: Any,
    paper_value: float,
    chance: float,
    note: str,
) -> Dict[str, Any]:
    try:
        value = float(our_value)
    except (TypeError, ValueError):
        value = math.nan
    return {
        "metric": metric,
        "our_split": "test (best validation checkpoint)",
        "our_value": value,
        "paper_split": "validation",
        "paper_value": paper_value,
        "delta_percentage_points": (value - paper_value) * 100
        if math.isfinite(value)
        else math.nan,
        "chance": chance,
        "paper": "Chen et al., Decoding EEG Speech Perception with Transformers and VAE-based Data Augmentation",
        "note": note,
    }


def generate_article_comparison(run_dir: Path) -> Dict[str, Any]:
    """Compare the word-classification run with the attached Alice paper."""

    final_results = _read_json(run_dir / "final_results.json")
    metrics = final_results.get("test_metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("final_results.json does not contain an object test_metrics")

    rows = [
        _metric_row(
            metric="Top-1 accuracy, 601-word retrieval vocabulary",
            our_value=metrics.get("top1_accuracy_retrieval601"),
            paper_value=ARTICLE_TOP1,
            chance=1.0 / ARTICLE_RETRIEVAL_SIZE,
            note="Directly comparable vocabulary size; our value is test, paper reports validation.",
        ),
        _metric_row(
            metric="Top-10 accuracy, 601-word retrieval vocabulary",
            our_value=metrics.get("top10_accuracy_retrieval601"),
            paper_value=ARTICLE_TOP10,
            chance=10.0 / ARTICLE_RETRIEVAL_SIZE,
            note="Directly comparable vocabulary size; our value is test, paper reports validation.",
        ),
    ]

    payload = {
        "experiment_name": final_results.get("experiment_name", run_dir.name),
        "run_dir": str(run_dir),
        "word_classifier_reference": {
            "vocabulary_size": ARTICLE_RETRIEVAL_SIZE,
            "validation_top1_accuracy": ARTICLE_TOP1,
            "validation_top10_accuracy": ARTICLE_TOP10,
            "inverse_frequency_weighted_top10": "<0.001",
        },
        "sequence_to_sequence_reference": [
            {
                "setting": "22 subjects, 200 epochs",
                "train_loss": 0.35,
                "train_wer": 0.2097,
                "validation_wer": 0.9248,
            },
            {
                "setting": "22 subjects, masking, 200 epochs",
                "train_loss": 0.50,
                "train_wer": 0.3773,
                "validation_wer": 0.9232,
            },
            {
                "setting": "10 subjects, stratified split, masking, 800 epochs",
                "train_loss": 0.06,
                "train_wer": 0.0291,
                "validation_wer": 0.9323,
            },
        ],
        "vae_reference": (
            "Replacing 50% or 90% of training samples with VAE-generated EEG "
            "did not improve validation performance."
        ),
        "scope_note": (
            "The current MEG-XL downstream task is word retrieval classification. "
            "Seq2Seq WER and VAE augmentation are recorded as contextual baselines, "
            "not treated as directly comparable outputs."
        ),
        "comparisons": rows,
    }

    _write_json(run_dir / "alice_reference_comparison.json", payload)
    _write_rows_csv(run_dir / "alice_reference_comparison.csv", rows)
    _write_rows_markdown(run_dir / "alice_reference_comparison.md", rows)
    return payload


def generate_alice_combined_report() -> Dict[str, Any] | None:
    specification = os.environ.get("ALICE_COMPARISON_RUNS", "").strip()
    if not specification:
        return None
    output = os.environ.get("ALICE_COMPARISON_OUTPUT", "").strip()
    if not output:
        raise ValueError(
            "ALICE_COMPARISON_OUTPUT is required with ALICE_COMPARISON_RUNS"
        )

    run_specs = parse_comparison_runs(specification)
    output_dir = Path(output)
    manifest = generate_comparison_report(
        run_specs,
        output_dir,
        retrieval_sizes=(50, 250),
        top_k=10,
    )

    # The reusable reporter predates Alice and retains this legacy filename.
    # Keep it for compatibility and add dataset-correct aliases.
    for suffix in ("csv", "md"):
        source = output_dir / f"weissbart_three_way_test_metrics.{suffix}"
        destination = output_dir / f"alice_three_way_test_metrics.{suffix}"
        if source.exists():
            shutil.copy2(source, destination)

    reference_rows: List[Dict[str, Any]] = []
    reference_runs = []
    for label, run_dir in run_specs:
        comparison_path = Path(run_dir) / "alice_reference_comparison.json"
        comparison = _read_json(comparison_path)
        reference_runs.append({"model": label, "path": str(comparison_path)})
        for row in comparison.get("comparisons", []):
            reference_rows.append({"model": label, **row})

    _write_rows_csv(
        output_dir / "alice_reference_three_way_comparison.csv", reference_rows
    )
    _write_rows_markdown(
        output_dir / "alice_reference_three_way_comparison.md", reference_rows
    )
    _write_json(
        output_dir / "alice_reference_three_way_comparison.json",
        {
            "runs": reference_runs,
            "reference_validation_top1_601": ARTICLE_TOP1,
            "reference_validation_top10_601": ARTICLE_TOP10,
            "comparisons": reference_rows,
        },
    )

    manifest["alice_long_metrics_csv"] = str(
        output_dir / "alice_three_way_test_metrics.csv"
    )
    manifest["alice_reference_comparison_csv"] = str(
        output_dir / "alice_reference_three_way_comparison.csv"
    )
    _write_json(output_dir / "alice_megxl_report_manifest.json", manifest)
    return manifest


def main() -> Any:
    evaluator.get_dataset_class = get_dataset_class
    evaluator.get_default_max_channel_dim = get_default_max_channel_dim
    evaluator.get_dataset_extra_kwargs = get_dataset_extra_kwargs
    evaluator.get_num_sensor_types_for_config = get_num_sensor_types_for_config

    optimized._evaluate_epoch = _alice_evaluate_epoch
    optimized.install_optimized_word_finetuning(evaluator)

    if not any(
        argument == "--config-name" or argument.startswith("--config-name=")
        for argument in sys.argv[1:]
    ):
        sys.argv.insert(
            1,
            "--config-name=eval_criss_cross_word_classification_alice_eeg",
        )

    result = evaluator.main()

    save_dir_override = _cli_override("logging.save_dir")
    if not save_dir_override:
        raise ValueError(
            "logging.save_dir must be supplied so reports can locate final_results.json"
        )
    run_dir = Path(save_dir_override)

    print(f"\nGenerating MEG-XL metrics and figures in {run_dir}...")
    generate_run_report(run_dir, retrieval_sizes=(50, 250), top_k=10)

    print("Generating comparison with the attached Alice EEG paper...")
    generate_article_comparison(run_dir)

    combined = generate_alice_combined_report()
    if combined is not None:
        print("Generated combined Alice three-way report:")
        for figure in combined.get("figures", []):
            print(f"  Figure: {figure}")

    return result


if __name__ == "__main__":
    main()
