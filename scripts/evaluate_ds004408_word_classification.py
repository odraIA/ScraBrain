"""ds004408 fine-tuning entry point with tier-safe word alignment and reporting."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig

from brainstorm import evaluate_criss_cross_word_classification as evaluator
from brainstorm.data.openneuroEEG_ds004408_word_aligned_dataset import (
    OpenNeuroEEGDs004408WordAlignedDataset,
)
from brainstorm.megxl_test_reporting import (
    generate_run_report,
    maybe_generate_comparison_from_environment,
)
from brainstorm.models.eeg_sensor_embedding_transformer import (
    EEGSensorEmbeddingCrissCrossTransformerModule,
)
from brainstorm.optimized_word_finetuning import install_optimized_word_finetuning


_BASE_GET_DATASET_CLASS = evaluator.get_dataset_class
_BASE_GET_DEFAULT_MAX_CHANNEL_DIM = evaluator.get_default_max_channel_dim
_BASE_GET_DATASET_EXTRA_KWARGS = evaluator.get_dataset_extra_kwargs
_BASE_GET_NUM_SENSOR_TYPES = evaluator.get_num_sensor_types_for_config
_BASE_GENERATE_WORD_EMBEDDINGS = evaluator.generate_word_embeddings


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
            "eeg_sensor_type": cfg.data.get("eeg_sensor_type", "eeg"),
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
            cfg.data.get("eeg_sensor_type", "eeg")
        )
        embedding_type_id = int(cfg.model.get("eeg_sensor_embedding_type_id", 2))
        num_sensor_types = max(
            num_sensor_types,
            sensor_type_id + 1,
            embedding_type_id + 1,
        )
    return num_sensor_types


def generate_word_embeddings_locked(
    vocab: List[str],
    vocab_size: Optional[int] = None,
    layer: int = 12,
    cache_dir: str = "./embeddings_cache",
    device: str = "cpu",
    verbose: bool = True,
    dataset_type: str = "armeni",
) -> torch.Tensor:
    """Serialize shared T5-cache creation across parallel ds004408 runs."""

    resolved_vocab_size = len(vocab) if vocab_size is None else int(vocab_size)
    vocab_hash = hashlib.sha256(" ".join(sorted(vocab)).encode()).hexdigest()[:8]
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / (
        f"word_embeddings_{dataset_type}_{resolved_vocab_size}_"
        f"layer{layer}_{vocab_hash}.pt"
    )
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return _BASE_GENERATE_WORD_EMBEDDINGS(
            vocab=vocab,
            vocab_size=resolved_vocab_size,
            layer=layer,
            cache_dir=str(cache_root),
            device=device,
            verbose=verbose,
            dataset_type=dataset_type,
        )


def _cli_override(name: str) -> str | None:
    prefixes = (f"{name}=", f"+{name}=", f"++{name}=")
    for argument in reversed(sys.argv[1:]):
        for prefix in prefixes:
            if argument.startswith(prefix):
                return argument[len(prefix):].strip().strip('"').strip("'")
    return None


def _configure_eeg_sensor_embedding() -> int:
    """Keep ds004408 physically EEG while selecting the checkpoint's lookup row."""

    configured = _cli_override("model.eeg_sensor_embedding_type_id")
    if configured is None:
        configured = os.environ.get("EEG_SENSOR_EMBEDDING_TYPE_ID", "2")
    embedding_type_id = int(configured)
    if embedding_type_id not in (1, 2):
        raise ValueError(
            "ds004408 fine-tuning supports EEG embedding rows 1 or 2; "
            f"got {embedding_type_id}. The physical sensor type remains EEG=2."
        )

    os.environ["EEG_SENSOR_EMBEDDING_TYPE_ID"] = str(embedding_type_id)
    evaluator.CrissCrossTransformerModule = (
        EEGSensorEmbeddingCrissCrossTransformerModule
    )
    print(
        "ds004408 sensor configuration: physical EEG type=2; "
        f"sensor embedding lookup row={embedding_type_id}"
    )
    return embedding_type_id


def main() -> Any:
    _configure_eeg_sensor_embedding()
    evaluator.get_dataset_class = get_dataset_class
    evaluator.get_default_max_channel_dim = get_default_max_channel_dim
    evaluator.get_dataset_extra_kwargs = get_dataset_extra_kwargs
    evaluator.get_num_sensor_types_for_config = get_num_sensor_types_for_config
    evaluator.generate_word_embeddings = generate_word_embeddings_locked
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
