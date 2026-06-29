"""ds004408 fine-tuning entry point with tier-safe word alignment and reporting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig

from brainstorm import evaluate_criss_cross_word_classification as evaluator
from brainstorm.data.openneuroEEG_ds004408_word_aligned_dataset import (
    OpenNeuroEEGDs004408WordAlignedDataset,
)
from brainstorm.megxl_test_reporting import (
    generate_run_report,
    maybe_generate_comparison_from_environment,
)
from brainstorm.optimized_word_finetuning import install_optimized_word_finetuning


_BASE_GET_DATASET_CLASS = evaluator.get_dataset_class
_BASE_GET_DEFAULT_MAX_CHANNEL_DIM = evaluator.get_default_max_channel_dim
_BASE_GET_DATASET_EXTRA_KWARGS = evaluator.get_dataset_extra_kwargs
_BASE_GET_NUM_SENSOR_TYPES = evaluator.get_num_sensor_types_for_config


def get_dataset_class(dataset_type: str):
    if dataset_type == "openneuro_ds004408":
        return OpenNeuroEEGDs004408WordAlignedDataset
    return _BASE_GET_DATASET_CLASS(dataset_type)


def get_default_max_channel_dim(dataset_type: str) -> int:
    if dataset_type == "openneuro_ds004408":
        return 128
    return _BASE_GET_DEFAULT_MAX_CHANNEL_DIM(dataset_type)


def get_dataset_extra_kwargs(dataset_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if dataset_type == "openneuro_ds004408":
        return {
            "dataset_name": cfg.data.get("dataset_name", "openneuroEEG_ds004408"),
            "task_mode": cfg.data.get("task_mode", "listening"),
            "tokenizer_name": cfg.model.get("tokenizer_name", "biocodec"),
            "eeg_sensor_type": cfg.data.get("eeg_sensor_type", "grad"),
            "word_tier_names": list(cfg.data.get("word_tier_names", ["word", "words"])),
            "montage_name": cfg.data.get("montage_name", "biosemi128"),
            "drop_bad_channels": bool(cfg.data.get("drop_bad_channels", True)),
            "cache_version": cfg.data.get("cache_version", "ds004408_word_aligned_v2"),
            "allow_missing_word_alignment": False,
        }
    return _BASE_GET_DATASET_EXTRA_KWARGS(dataset_type, cfg)


def get_num_sensor_types_for_config(cfg: DictConfig) -> int:
    num_sensor_types = _BASE_GET_NUM_SENSOR_TYPES(cfg)
    if cfg.data.get("dataset_type") == "openneuro_ds004408":
        sensor_type_id = evaluator.resolve_sensor_type_id(
            cfg.data.get("eeg_sensor_type", "grad")
        )
        num_sensor_types = max(num_sensor_types, sensor_type_id + 1)
    return num_sensor_types


def _cli_override(name: str) -> str | None:
    prefix = f"{name}="
    for argument in reversed(sys.argv[1:]):
        if argument.startswith(prefix):
            return argument[len(prefix):].strip().strip('"').strip("'")
    return None


def main() -> Any:
    evaluator.get_dataset_class = get_dataset_class
    evaluator.get_default_max_channel_dim = get_default_max_channel_dim
    evaluator.get_dataset_extra_kwargs = get_dataset_extra_kwargs
    evaluator.get_num_sensor_types_for_config = get_num_sensor_types_for_config
    install_optimized_word_finetuning(evaluator)

    if not any(
        argument == "--config-name" or argument.startswith("--config-name=")
        for argument in sys.argv[1:]
    ):
        sys.argv.insert(1, "--config-name=ds004408_word_finetuning")

    result = evaluator.main()

    save_dir_override = _cli_override("logging.save_dir")
    if not save_dir_override:
        raise ValueError(
            "logging.save_dir must be supplied so the automatic MEG-XL report "
            "can locate final_results.json"
        )

    run_dir = Path(save_dir_override)
    print(f"\nGenerating MEG-XL paper metrics and figures in {run_dir}...")
    manifest = generate_run_report(run_dir)
    print(f"Paper report manifest: {run_dir / 'paper_report_manifest.json'}")
    for figure in manifest.get("figures", []):
        print(f"  Figure: {figure}")

    comparison_manifest = maybe_generate_comparison_from_environment()
    if comparison_manifest is not None:
        print("\nGenerated combined ds004408 / MEG-XL comparison report:")
        for figure in comparison_manifest.get("figures", []):
            print(f"  Figure: {figure}")

    return result


if __name__ == "__main__":
    main()
