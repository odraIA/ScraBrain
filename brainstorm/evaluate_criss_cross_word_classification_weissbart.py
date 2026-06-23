"""Fine-tuning entry point for the Weissbart EEG word-aligned dataset.

This keeps the generic CrissCross word-classification implementation unchanged
and only registers the Weissbart dataset before invoking its Hydra entry point.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from omegaconf import DictConfig

from brainstorm import evaluate_criss_cross_word_classification as evaluator
from brainstorm.data.weissbart_eeg_word_aligned_dataset import (
    WeissbartEEGWordAlignedDataset,
)


_BASE_GET_DATASET_CLASS = evaluator.get_dataset_class
_BASE_GET_DEFAULT_MAX_CHANNEL_DIM = evaluator.get_default_max_channel_dim
_BASE_GET_DATASET_EXTRA_KWARGS = evaluator.get_dataset_extra_kwargs
_BASE_GET_NUM_SENSOR_TYPES = evaluator.get_num_sensor_types_for_config


def get_dataset_class(dataset_type: str):
    if dataset_type == "weissbart_eeg":
        return WeissbartEEGWordAlignedDataset
    return _BASE_GET_DATASET_CLASS(dataset_type)


def get_default_max_channel_dim(dataset_type: str) -> int:
    if dataset_type == "weissbart_eeg":
        return 64
    return _BASE_GET_DEFAULT_MAX_CHANNEL_DIM(dataset_type)


def get_dataset_extra_kwargs(dataset_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if dataset_type == "weissbart_eeg":
        return {
            "eeg_sensor_type": cfg.data.get("eeg_sensor_type", "grad"),
            "dataset_name": cfg.data.get("dataset_name", "weissbart_eeg"),
            "task_mode": cfg.data.get("task_mode", "listening"),
            "tokenizer_name": cfg.model.get("tokenizer_name", "biocodec"),
        }
    return _BASE_GET_DATASET_EXTRA_KWARGS(dataset_type, cfg)


def get_num_sensor_types_for_config(cfg: DictConfig) -> int:
    num_sensor_types = _BASE_GET_NUM_SENSOR_TYPES(cfg)
    if cfg.data.get("dataset_type") == "weissbart_eeg":
        sensor_type_id = evaluator.resolve_sensor_type_id(
            cfg.data.get("eeg_sensor_type", "grad")
        )
        num_sensor_types = max(num_sensor_types, sensor_type_id + 1)
    return num_sensor_types


def main() -> Any:
    evaluator.get_dataset_class = get_dataset_class
    evaluator.get_default_max_channel_dim = get_default_max_channel_dim
    evaluator.get_dataset_extra_kwargs = get_dataset_extra_kwargs
    evaluator.get_num_sensor_types_for_config = get_num_sensor_types_for_config

    if not any(
        argument == "--config-name" or argument.startswith("--config-name=")
        for argument in sys.argv[1:]
    ):
        sys.argv.insert(
            1,
            "--config-name=eval_criss_cross_word_classification_weissbart_eeg",
        )
    return evaluator.main()


if __name__ == "__main__":
    main()
