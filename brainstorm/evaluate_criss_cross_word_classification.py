"""
Word classification evaluation for CrissCrossTransformer.

Evaluates on multiple subjects with session-based temporal split:
- Train: ses-001 through ses-008
- Val: ses-009
- Test: ses-010

Training: Uses ALL unique words (no vocabulary restriction, no OOV during training).

Evaluation: Reports top-10 accuracy when retrieving from subsets of most frequent words.
For a retrieval set of size K (e.g., top-50 or top-250):
- Only samples whose true label is in the top-K words are evaluated
- Retrieval is performed against the top-K word embeddings
- Samples with labels outside the top-K are skipped

Metrics:
- Top-10 retrieval accuracy for each retrieval set size
- Balanced top-10 accuracy (macro-averaged across retrieval set)
- Embedding quality metrics (cosine similarity, norms)

Usage:
    python -m brainstorm.evaluate_criss_cross_word_classification \
        model.criss_cross_checkpoint=path/to/ckpt.ckpt \
        training.batch_size=4 \
        training.num_epochs=50
"""

import logging
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import pytorch_lightning as pl
from transformers import T5EncoderModel, T5Tokenizer
import wandb
from tqdm import tqdm
import pandas as pd

from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.neuro_tokenizers.factory import NeuroTokenizerAdapter, load_neuro_tokenizer
from brainstorm.data.armeni_word_aligned_dataset import ArmeniWordAlignedDataset
from brainstorm.data.gwilliams_word_aligned_dataset import GwilliamsWordAlignedDataset
from brainstorm.data.libribrain_word_aligned_dataset import LibriBrainWordAlignedDataset
from brainstorm.data.zuco_word_aligned_dataset import ZuCoWordAlignedDataset
from brainstorm.data.eeg_word_aligned_dataset import (
    EEGDashWordAlignedDataset,
    OpenNeuroEEGWordAlignedDataset,
    PooledWordAlignedDataset,
    scan_bids_eeg_channel_counts,
    scan_zuco_channel_counts,
)
from brainstorm.eval_metrics_history import append_epoch_metrics_history, resolve_checkpoint_dir
from brainstorm.losses.contrastive import SigLipLoss

logger = logging.getLogger(__name__)


class TeeStream:
    """Mirror writes to the original stream and a run log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())


def _install_run_tee(save_dir: Path) -> Any:
    log_path = save_dir / "stdout_stderr.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_file


def _git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8").strip()
    except Exception:
        return None


def _command_line() -> List[str]:
    return [sys.executable, *sys.argv]


def _training_mode(cfg: DictConfig, init_mode: Optional[str] = None) -> str:
    if init_mode:
        return init_mode
    if cfg.model.get("train_from_scratch", False):
        return "scratch"
    if cfg.model.get("use_promoted_checkpoint", False):
        return "promoted"
    return "pretrained"


def _dataset_entries_for_metadata(cfg: DictConfig) -> List[Dict[str, Any]]:
    entries = cfg.data.get("datasets", None)
    if entries is None:
        entries = [cfg.data]
    datasets: List[Dict[str, Any]] = []
    for entry in entries:
        datasets.append({
            "dataset_type": str(entry.get("dataset_type", cfg.data.get("dataset_type", ""))),
            "dataset_name": str(entry.get("dataset_name", entry.get("dataset_type", cfg.data.get("dataset_type", "")))),
            "task_mode": str(entry.get("task_mode", cfg.data.get("task_mode", ""))),
            "root": str(entry.get("root", cfg.data.get("root", ""))),
            "tasks": list(entry.get("tasks", cfg.data.get("tasks", [])) or []),
        })
    return datasets


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_resolved_config_snapshot(save_dir: Path, cfg: DictConfig) -> Path:
    path = save_dir / "config_resolved.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def write_run_metadata(
    *,
    save_dir: Path,
    checkpoint_dir: Path,
    cfg: DictConfig,
    init_mode: str,
    init_checkpoint: str,
    architecture_checkpoint: Optional[str],
    split_sizes: Dict[str, int],
    channel_count: int,
    tokenizer: NeuroTokenizerAdapter,
    config_snapshot: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = save_dir / "run_metadata.json"
    payload: Dict[str, Any] = {
        "experiment_name": str(cfg.logging.get("experiment_name", save_dir.name)),
        "datasets": _dataset_entries_for_metadata(cfg),
        "dataset_type": str(cfg.data.get("dataset_type", "")),
        "task_mode": str(cfg.data.get("task_mode", "")),
        "target_sfreq": float(cfg.data.get("target_sfreq", 0.0)),
        "tokenizer": {
            "name": str(cfg.model.get("tokenizer_name", "biocodec")),
            "checkpoint": str(cfg.model.get("tokenizer_checkpoint", "")),
            "downsample_ratio": int(getattr(tokenizer, "downsample_ratio", 0)),
            "n_q": int(getattr(tokenizer, "n_q", 0)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        },
        "training_mode": _training_mode(cfg, init_mode),
        "train_from_scratch": bool(cfg.model.get("train_from_scratch", False)),
        "use_promoted_checkpoint": bool(cfg.model.get("use_promoted_checkpoint", False)),
        "checkpoint_paths": {
            "initial_checkpoint": str(init_checkpoint),
            "architecture_checkpoint": str(architecture_checkpoint or ""),
            "promoted_checkpoint": str(cfg.model.get("promoted_checkpoint", "") or ""),
            "checkpoint_latest": str(checkpoint_dir / "checkpoint_latest.pt"),
            "checkpoint_best": str(checkpoint_dir / "checkpoint_best.pt"),
        },
        "artifact_paths": {
            "save_dir": str(save_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "stdout_stderr_log": str(save_dir / "stdout_stderr.log"),
            "epoch_metrics_csv": str(save_dir / "epoch_metrics.csv"),
            "epoch_metrics_jsonl": str(save_dir / "epoch_metrics.jsonl"),
            "final_results_txt": str(save_dir / "final_results.txt"),
            "final_results_json": str(save_dir / "final_results.json"),
            "config_snapshot": str(config_snapshot),
        },
        "split_sizes": split_sizes,
        "channel_count": int(channel_count),
        "command": _command_line(),
        "timestamp": datetime_now_iso(),
        "git_hash": _git_hash(),
        "seed": int(cfg.seed),
    }
    if extra:
        payload.update(extra)
    _write_json(path, payload)
    return path


def datetime_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Dataset Factory Functions
# ============================================================================

def get_dataset_class(dataset_type: str):
    """
    Get the appropriate dataset class based on dataset_type.

    Args:
        dataset_type: One of "armeni", "gwilliams", "libribrain", or "zuco"

    Returns:
        Dataset class to instantiate
    """
    dataset_classes = {
        "armeni": ArmeniWordAlignedDataset,
        "gwilliams": GwilliamsWordAlignedDataset,
        "libribrain": LibriBrainWordAlignedDataset,
        "zuco": ZuCoWordAlignedDataset,
        "eegdash": EEGDashWordAlignedDataset,
        "openneuro_eeg": OpenNeuroEEGWordAlignedDataset,
        "openneuro_ds004408": OpenNeuroEEGWordAlignedDataset,
        "openneuro_ds007808": OpenNeuroEEGWordAlignedDataset,
    }

    if dataset_type not in dataset_classes:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Must be one of: {list(dataset_classes.keys())}"
        )

    return dataset_classes[dataset_type]


def get_default_max_channel_dim(dataset_type: str) -> int:
    """
    Get the default max_channel_dim for each dataset type.

    - Armeni: 306 MEG channels (CTF system)
    - Gwilliams: 208 MEG channels (KIT/Ricoh system)
    - LibriBrain: 306 MEG channels (Elekta Neuromag system)
    - ZuCo: 105 EEG channels (HydroCel Geodesic Sensor Net)
    """
    defaults = {
        "armeni": 306,
        "gwilliams": 208,
        "libribrain": 306,
        "zuco": 105,
        "eegdash": 128,
        "openneuro_eeg": 128,
        "openneuro_ds004408": 128,
        "openneuro_ds007808": 128,
        "eeg_pooled": 128,
    }
    return defaults.get(dataset_type, 306)


def resolve_sensor_type_id(sensor_type: str) -> int:
    aliases = {
        "grad": 0,
        "gradiometer": 0,
        "mag": 1,
        "meg": 1,
        "magnetometer": 1,
        "eeg": 2,
    }
    key = str(sensor_type).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unknown sensor type {sensor_type!r}. "
            f"Expected one of: {sorted(aliases.keys())}"
        )
    return aliases[key]


def get_num_sensor_types_for_config(cfg: DictConfig) -> int:
    num_sensor_types = int(cfg.model.get("num_sensor_types", 2))
    dataset_type = cfg.data.get("dataset_type", "armeni")
    has_eeg_dataset = dataset_type in {
        "zuco",
        "eegdash",
        "openneuro_eeg",
        "openneuro_ds004408",
        "openneuro_ds007808",
        "eeg_pooled",
    } or cfg.data.get("datasets") is not None
    if has_eeg_dataset:
        eeg_sensor_type = cfg.data.get("eeg_sensor_type", "grad")
        num_sensor_types = max(num_sensor_types, resolve_sensor_type_id(eeg_sensor_type) + 1)
    return num_sensor_types


def get_dataset_extra_kwargs(dataset_type: str, cfg: DictConfig) -> Dict[str, Any]:
    if dataset_type == "zuco":
        return {
            "eeg_sensor_type": cfg.data.get("eeg_sensor_type", "grad"),
            "dataset_name": cfg.data.get("dataset_name", "zuco"),
            "task_mode": cfg.data.get("task_mode", "reading"),
            "tokenizer_name": cfg.model.get("tokenizer_name", "biocodec"),
        }
    if dataset_type in {"eegdash", "openneuro_eeg", "openneuro_ds004408", "openneuro_ds007808"}:
        return {
            "dataset_name": cfg.data.get("dataset_name", dataset_type),
            "task_mode": cfg.data.get("task_mode", "reading" if dataset_type == "eegdash" else "listening"),
            "tokenizer_name": cfg.model.get("tokenizer_name", "biocodec"),
        }
    return {}


def _as_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if hasattr(value, "__iter__"):
        return [str(item) for item in value]
    return [str(value)]


def _dataset_config_get(dataset_cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(dataset_cfg, DictConfig):
        return dataset_cfg.get(key, default)
    if isinstance(dataset_cfg, dict):
        return dataset_cfg.get(key, default)
    return default


def _is_auto_channel_dim(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _scan_dataset_max_channel_dim(dataset_cfg: Any) -> Tuple[Optional[int], List[str]]:
    dataset_type = _dataset_config_get(dataset_cfg, "dataset_type", "")
    root = _dataset_config_get(dataset_cfg, "root")
    tasks = _as_list(_dataset_config_get(dataset_cfg, "tasks", None))
    if root is None:
        return None, [f"{dataset_type or 'dataset'} has no root configured"]

    if dataset_type == "zuco":
        counts = scan_zuco_channel_counts(root)
    elif dataset_type in {"eegdash", "openneuro_eeg", "openneuro_ds004408", "openneuro_ds007808"}:
        counts = scan_bids_eeg_channel_counts(root, tasks=tasks)
    else:
        return None, [f"{dataset_type or 'dataset'} does not support automatic EEG channel scanning"]

    if not counts:
        return None, [f"{dataset_type} at {root} had no readable EEG channel counts"]

    max_count = max(item.n_channels for item in counts)
    max_items = [item for item in counts if item.n_channels == max_count]
    details = [
        f"{dataset_type} max={max_count} channels across {len(counts)} recording(s); "
        f"example={max_items[0].path} ({max_items[0].method})"
    ]
    return max_count, details


def resolve_max_channel_dim(cfg: DictConfig) -> int:
    dataset_type = cfg.data.get("dataset_type", "armeni")
    configured = cfg.data.get("max_channel_dim", get_default_max_channel_dim(dataset_type))

    dataset_entries = cfg.data.get("datasets", None)
    needs_auto = _is_auto_channel_dim(configured)
    if dataset_entries is not None:
        needs_auto = needs_auto or any(_is_auto_channel_dim(_dataset_config_get(entry, "max_channel_dim", configured)) for entry in dataset_entries)

    if not needs_auto:
        return int(configured)

    entries = list(dataset_entries) if dataset_entries is not None else [cfg.data]
    resolved_values: List[int] = []
    details: List[str] = []
    for entry in entries:
        entry_value = _dataset_config_get(entry, "max_channel_dim", configured)
        if not _is_auto_channel_dim(entry_value) and entry_value is not None:
            resolved_values.append(int(entry_value))
            continue
        max_count, entry_details = _scan_dataset_max_channel_dim(entry)
        details.extend(entry_details)
        if max_count is not None:
            resolved_values.append(max_count)

    if not resolved_values:
        raise ValueError(
            "Could not resolve data.max_channel_dim=auto. "
            "Set an integer max_channel_dim or check that EEG dataset roots are readable. "
            f"Scan details: {details}"
        )

    resolved = max(resolved_values)
    cfg.data.max_channel_dim = resolved
    if dataset_entries is not None:
        for entry in cfg.data.datasets:
            entry.max_channel_dim = resolved
    for detail in details:
        logger.info(f"Channel scan: {detail}")
    logger.info(f"Resolved max_channel_dim=auto to {resolved}")
    return resolved


def instantiate_word_dataset(
    cfg: DictConfig,
    sessions: Optional[List[str]],
    data_override: Optional[Any] = None,
):
    data_cfg = data_override if data_override is not None else cfg.data
    dataset_type = _dataset_config_get(data_cfg, "dataset_type", cfg.data.get("dataset_type", "armeni"))
    DatasetClass = get_dataset_class(dataset_type)
    max_channel_dim = _dataset_config_get(
        data_cfg,
        "max_channel_dim",
        cfg.data.get("max_channel_dim", get_default_max_channel_dim(dataset_type)),
    )
    if _is_auto_channel_dim(max_channel_dim):
        max_channel_dim = resolve_max_channel_dim(cfg)

    extra_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if data_override is not None:
        extra_cfg.data.dataset_name = _dataset_config_get(data_cfg, "dataset_name", dataset_type)
        extra_cfg.data.task_mode = _dataset_config_get(data_cfg, "task_mode", cfg.data.get("task_mode", ""))
        extra_cfg.data.eeg_sensor_type = _dataset_config_get(data_cfg, "eeg_sensor_type", cfg.data.get("eeg_sensor_type", "eeg"))
    dataset_extra_kwargs = get_dataset_extra_kwargs(dataset_type, extra_cfg)

    return DatasetClass(
        data_root=_dataset_config_get(data_cfg, "root", cfg.data.get("root")),
        subjects=_as_list(_dataset_config_get(data_cfg, "subjects", cfg.data.get("subjects"))),
        sessions=_as_list(_dataset_config_get(data_cfg, "sessions", sessions)),
        tasks=_as_list(_dataset_config_get(data_cfg, "tasks", cfg.data.get("tasks"))),
        segment_length=cfg.data.segment_length,
        subsegment_duration=cfg.data.subsegment_duration,
        words_per_segment=cfg.data.words_per_segment,
        window_onset_offset=cfg.data.window_onset_offset,
        cache_dir=_dataset_config_get(data_cfg, "cache_dir", cfg.data.cache_dir),
        l_freq=cfg.data.l_freq,
        h_freq=cfg.data.h_freq,
        target_sfreq=cfg.data.target_sfreq,
        max_channel_dim=max_channel_dim,
        **dataset_extra_kwargs,
    )


def create_word_dataset(cfg: DictConfig, sessions: Optional[List[str]] = None):
    dataset_entries = cfg.data.get("datasets", None)
    if dataset_entries is None:
        return instantiate_word_dataset(cfg, sessions=sessions)

    datasets = []
    for entry in dataset_entries:
        try:
            datasets.append(instantiate_word_dataset(cfg, sessions=sessions, data_override=entry))
        except (FileNotFoundError, ValueError) as exc:
            if _dataset_config_get(entry, "allow_missing", False):
                logger.warning(f"Skipping optional dataset {_dataset_config_get(entry, 'dataset_name', entry)}: {exc}")
                continue
            raise

    return PooledWordAlignedDataset(datasets)


def get_dataset_words(dataset: Any, idx: int) -> List[str]:
    if hasattr(dataset, "get_segment_words"):
        return list(dataset.get_segment_words(idx))
    if isinstance(dataset, torch.utils.data.Subset):
        return get_dataset_words(dataset.dataset, dataset.indices[idx])
    rec_idx, group_idx = dataset.segment_index[idx]
    return [word["word"] for word in dataset.word_groups[rec_idx][group_idx]]


def get_dataset_split_group(dataset: Any, idx: int, group_kind: str) -> str:
    if hasattr(dataset, "get_split_group"):
        return str(dataset.get_split_group(idx, group_kind))
    if isinstance(dataset, torch.utils.data.Subset):
        return get_dataset_split_group(dataset.dataset, dataset.indices[idx], group_kind)
    if group_kind == "sentence":
        return " ".join(get_dataset_words(dataset, idx))
    rec_idx, group_idx = dataset.segment_index[idx]
    rec = dataset.recordings[rec_idx]
    dataset_name = getattr(dataset, "dataset_name", type(dataset).__name__)
    subject = rec.get("subject", "")
    session = rec.get("session", "")
    task = rec.get("task", "")
    if group_kind == "subject":
        return f"{dataset_name}:{subject}"
    if group_kind == "session":
        return f"{dataset_name}:{subject}:{session}"
    return f"{dataset_name}:{subject}:{session}:{task}:{rec_idx}"


def hash_key_to_split(key: str, split_ratios: List[float], seed: int = 42) -> str:
    salted = f"{seed}:{key}"
    hash_obj = hashlib.sha256(salted.encode("utf-8"))
    hash_float = (int(hash_obj.hexdigest(), 16) % 1_000_000) / 1_000_000.0
    cumsum = 0.0
    for split_name, ratio in zip(["train", "val", "test"], split_ratios):
        cumsum += ratio
        if hash_float < cumsum:
            return split_name
    return "test"


def assign_no_leak_splits(dataset: Any, cfg: DictConfig) -> Tuple[List[int], List[int], List[int], Dict[int, str]]:
    preference = cfg.data.get("split_group", "sentence")
    candidates = ["sentence"] if preference == "sentence" else [preference]
    if preference == "auto":
        candidates = ["subject", "session", "recording", "sentence"]

    best_result = None
    for group_kind in candidates:
        groups: Dict[str, List[int]] = {}
        idx_to_key: Dict[int, str] = {}
        for idx in range(len(dataset)):
            key = get_dataset_split_group(dataset, idx, group_kind)
            groups.setdefault(key, []).append(idx)
            idx_to_key[idx] = key

        split_indices = {"train": [], "val": [], "test": []}
        for key, indices in groups.items():
            split = hash_key_to_split(key, cfg.data.split_ratios, cfg.seed)
            split_indices[split].extend(indices)

        result = (
            split_indices["train"],
            split_indices["val"],
            split_indices["test"],
            idx_to_key,
            group_kind,
        )
        best_result = result
        if all(len(split_indices[name]) > 0 for name in ("train", "val", "test")):
            break

    assert best_result is not None
    train_indices, val_indices, test_indices, idx_to_key, group_kind = best_result
    logger.info(f"Using no-leak split group: {group_kind}")
    return train_indices, val_indices, test_indices, idx_to_key


def build_vocabulary_from_word_datasets(datasets: List[Any]) -> Tuple[List[str], Counter]:
    word_counter = Counter()
    for dataset in datasets:
        for idx in range(len(dataset)):
            word_counter.update(get_dataset_words(dataset, idx))
    vocab = [word for word, _ in word_counter.most_common()]
    return vocab, word_counter


# ============================================================================
# Model Components
# ============================================================================

class CrissCrossWordEmbeddingExtractor(nn.Module):
    """
    Extract word embeddings from CrissCrossTransformer features.

    Architecture per word subsegment:
    1. Extract features for subsegment time range: [C, ~62-63, 512]
    2. Mean pool over time: [C, 512]
    3. Flatten: [C * 512]
    4. MLP: [C * 512] -> [hidden_dim] -> [1024]
    """

    def __init__(
        self,
        num_channels: int,
        latent_dim: int = 512,
        embed_dim: int = 1024,
        hidden_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        input_dim = num_channels * latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [C, T_subseg, latent_dim] - features for one word subsegment

        Returns:
            embedding: [embed_dim] - predicted T5 embedding
        """
        # Mean pool over time
        pooled = features.mean(dim=1)  # [C, latent_dim]

        # Flatten
        flat = pooled.reshape(-1)  # [C * latent_dim]

        # MLP projection
        embedding = self.mlp(flat)  # [embed_dim]

        return embedding


# ============================================================================
# Time Alignment Utilities
# ============================================================================

def map_raw_to_encoded_timesteps(
    start_sample: int,
    end_sample: int,
    downsample_ratio: int = 12
) -> Tuple[int, int]:
    """
    Map raw sample indices from subsegment_boundaries to encoded timesteps.

    Args:
        start_sample: Start index in raw samples (from subsegment_boundaries)
        end_sample: End index in raw samples
        downsample_ratio: Tokenizer downsampling ratio

    Returns:
        (start_t, end_t): Encoded timestep range
    """
    # Convert to encoded timesteps
    start_t = start_sample // downsample_ratio
    end_t = (end_sample + downsample_ratio - 1) // downsample_ratio  # Ceiling division

    return start_t, end_t


# ============================================================================
# Vocabulary Building
# ============================================================================

def build_vocabulary_from_dataset(
    data_root: Path,
    subject: str,
    sessions: List[str],
    task: str = "compr",
    top_k: int = 50
) -> List[str]:
    """
    Parse events.tsv files for specified subject/sessions and extract top-K words.

    This builds vocabulary ONLY from training sessions to avoid data leakage.
    Words outside this vocabulary will be excluded from the loss calculation.

    Args:
        data_root: Path to armeni2022 dataset
        subject: Subject ID (e.g., "sub-001")
        sessions: List of session IDs (e.g., ["ses-001", "ses-002", ...])
        task: Task name (default: "compr")
        top_k: Number of most frequent words to keep (default: 50)

    Returns:
        vocab: List[str] of top-K words sorted by frequency
    """
    logger.info(f"Building vocabulary from {subject} sessions: {sessions}")

    word_counts = Counter()

    # Parse events files for each session
    for session in sessions:
        events_path = data_root / subject / session / "meg" / f"{subject}_{session}_task-{task}_events.tsv"

        if not events_path.exists():
            logger.warning(f"Events file not found: {events_path}")
            continue

        # Load TSV
        df = pd.read_csv(events_path, sep='\t')

        # Filter to word_onset events
        word_events = df[df['type'].str.startswith('word_onset', na=False)]

        # Count word frequencies
        for word in word_events['value']:
            clean_word = str(word).strip('"').lower()
            if clean_word != 'sp':  # Skip silence markers
                word_counts[clean_word] += 1

    # Get top-K most frequent words
    most_common = word_counts.most_common(top_k)
    vocab = [word for word, count in most_common]

    logger.info(f"  Total unique words: {len(word_counts)}")
    logger.info(f"  Selected top-{top_k} words")
    logger.info(f"  Frequency range: {most_common[0][1]} to {most_common[-1][1]}")

    return vocab


def hash_sentence_to_split(
    words: List[str],
    split_ratios: List[float],
    seed: int = 42
) -> str:
    """
    Deterministically assign a sentence to train/val/test split based on its hash.

    The sentence is the concatenation of all words in the segment.
    Same sentence across different sessions/subjects will always get the same split.

    Args:
        words: List of words forming the sentence
        split_ratios: [train_ratio, val_ratio, test_ratio] summing to 1.0
        seed: Random seed for hash salt (default: 42)

    Returns:
        split: One of "train", "val", or "test"
    """
    # Concatenate words with space separator to form sentence
    sentence = " ".join(words)

    # Add seed salt for reproducibility
    salted_sentence = f"{seed}:{sentence}"

    # Hash using SHA256 for deterministic output
    hash_obj = hashlib.sha256(salted_sentence.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)

    # Map hash to [0, 1) range
    # Use modulo with large number to get uniform distribution
    hash_float = (hash_int % 1_000_000) / 1_000_000.0

    # Assign to split based on cumulative ratios
    cumsum = 0.0
    for split_name, ratio in zip(["train", "val", "test"], split_ratios):
        cumsum += ratio
        if hash_float < cumsum:
            return split_name

    # Fallback (should never reach here if ratios sum to 1.0)
    return "test"


# ============================================================================
# T5 Embedding Generation
# ============================================================================

def generate_word_embeddings(
    vocab: List[str],
    vocab_size: Optional[int] = None,
    layer: int = 12,
    cache_dir: str = './embeddings_cache',
    device: str = 'cpu',
    verbose: bool = True,
    dataset_type: str = 'armeni'
) -> torch.Tensor:
    """
    Generate or load cached T5 embeddings for vocabulary words.

    Args:
        vocab: List of words to generate embeddings for
        vocab_size: Vocabulary size for cache filename (if None, uses len(vocab))
        layer: Which T5 layer to extract embeddings from (default: 12)
        cache_dir: Directory to store cached embeddings
        device: Device to run T5 model on ('cpu' or 'cuda')
        verbose: Whether to print progress messages
        dataset_type: Dataset type for cache filename (default: 'armeni')

    Returns:
        Tensor of shape [vocab_size, embedding_dim] containing word embeddings
    """
    # Setup cache directory
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine vocab size
    if vocab_size is None:
        vocab_size = len(vocab)

    # Create a hash of the vocabulary words to ensure cache correctness
    # This prevents loading wrong embeddings when vocab changes (e.g., train-only vs full dataset)
    vocab_hash = hashlib.sha256(" ".join(sorted(vocab)).encode()).hexdigest()[:8]

    # Check for cached embeddings
    cache_path = cache_dir / f'word_embeddings_{dataset_type}_{vocab_size}_layer{layer}_{vocab_hash}.pt'

    if cache_path.exists():
        if verbose:
            logger.info(f"Loading cached word embeddings from: {cache_path}")
        embeddings = torch.load(cache_path, map_location='cpu', weights_only=False)
        if verbose:
            logger.info(f"  Loaded embeddings shape: {embeddings.shape}")
        return embeddings

    # Generate embeddings using T5
    if verbose:
        logger.info(f"Generating T5 word embeddings for {len(vocab)} words...")
        logger.info(f"  Using T5-large, layer {layer}")
        logger.info(f"  Device: {device}")

    # Load T5 model
    t5 = T5EncoderModel.from_pretrained('t5-large')
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    t5 = t5.to(device)
    t5.eval()

    embeddings = []
    with torch.no_grad():
        for i, word in enumerate(vocab):
            if verbose and (i % 10 == 0 or i == len(vocab) - 1):
                logger.info(f"  Processing word {i+1}/{len(vocab)}: '{word}'")

            # Convert to lowercase for consistency
            word = word.lower()

            # Tokenize
            tokens = tokenizer(word, return_tensors='pt', padding=True)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Forward pass
            outputs = t5(**tokens, output_hidden_states=True)

            # Extract hidden states from specified layer
            # Shape: (batch=1, seq_len, hidden_dim=1024)
            hidden_states = outputs.hidden_states[layer]

            # Ignore the last token (</s> end token)
            hidden_states = hidden_states[:, :-1, :]

            # Use mean pooling over token embeddings
            emb = hidden_states.mean(dim=1)  # Shape: (1, 1024)

            embeddings.append(emb.cpu())

    # Cleanup model
    del t5
    del tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Stack embeddings
    embeddings = torch.cat(embeddings, dim=0)  # Shape: [vocab_size, 1024]

    if verbose:
        logger.info(f"  Generated embeddings shape: {embeddings.shape}")
        logger.info(f"  Saving to cache: {cache_path}")

    # Save to cache
    torch.save(embeddings, cache_path)

    return embeddings


# ============================================================================
# Custom Collate Function
# ============================================================================

def create_word_level_collate_fn(word_to_idx: Dict[str, int]):
    """
    Create a collate function that tracks word labels for SigLIP loss.

    Args:
        word_to_idx: Mapping from word string to vocabulary index (includes ALL words)

    Returns:
        collate_fn: Collate function for DataLoader
    """
    def word_level_collate_fn(batch):
        """
        Collate function that expands 30s segments into individual word samples.

        All words are included since we train on the full vocabulary (no OOV filtering).
        Retrieval set filtering happens at evaluation time, not during collation.

        Input: List of dicts with keys:
            - meg: [C, 7500]
            - words: List[str] (10 words)
            - subsegment_boundaries: List[Dict] (10 boundaries)
            - sensor_xyzdir, sensor_types, sensor_mask

        Output: Dict with batched word samples:
            - meg: [B, C, 7500] - raw MEG for CrissCross
            - word_labels: [B*N] - word indices in vocabulary (N = words_per_segment * B)
            - subsegment_info: List of dicts with batch_idx, start, end for N words
            - sensor_xyzdir: [B, C, 6]
            - sensor_types: [B, C]
            - sensor_mask: [B, C]
        """
        batch_size = len(batch)

        # Pad MEG tensors to same length before stacking
        # (segments may have slightly different lengths after resampling)
        meg_tensors = [s['meg'] for s in batch]
        max_len = max(m.shape[-1] for m in meg_tensors)
        meg_padded = []
        for m in meg_tensors:
            if m.shape[-1] < max_len:
                pad_size = max_len - m.shape[-1]
                m = torch.nn.functional.pad(m, (0, pad_size), mode='constant', value=0)
            meg_padded.append(m)
        meg = torch.stack(meg_padded)
        sensor_xyzdir = torch.stack([s['sensor_xyzdir'] for s in batch])
        sensor_types = torch.stack([s['sensor_types'] for s in batch])
        sensor_mask = torch.stack([s['sensor_mask'] for s in batch])

        # Extract word labels and subsegment info
        word_labels = []
        subsegment_info = []

        for batch_idx, sample in enumerate(batch):
            for subseg_idx, (word, boundary) in enumerate(zip(sample['words'], sample['subsegment_boundaries'])):
                # Include all words - no OOV filtering since vocab contains all words
                if word not in word_to_idx:
                    # This should never happen if vocab is built correctly
                    logger.warning(f"Word '{word}' not in vocabulary - skipping")
                    continue

                word_labels.append(word_to_idx[word])
                subsegment_info.append({
                    'batch_idx': batch_idx,
                    'subseg_idx': subseg_idx,
                    'start_sample': boundary['start_sample'],
                    'end_sample': boundary['end_sample']
                })

        return {
            'meg': meg,
            'word_labels': torch.tensor(word_labels, dtype=torch.long),
            'subsegment_info': subsegment_info,
            'sensor_xyzdir': sensor_xyzdir,
            'sensor_types': sensor_types,
            'sensor_mask': sensor_mask
        }

    return word_level_collate_fn


# ============================================================================
# Checkpoint Loading
# ============================================================================

def load_tokenizer_from_config(cfg: DictConfig) -> NeuroTokenizerAdapter:
    tokenizer_name = cfg.model.get("tokenizer_name", "biocodec")
    tokenizer_checkpoint = cfg.model.get("tokenizer_checkpoint", None)
    logger.info(f"Loading tokenizer: name={tokenizer_name}, checkpoint={tokenizer_checkpoint}")
    tokenizer = load_neuro_tokenizer(
        tokenizer_name=tokenizer_name,
        checkpoint_path=tokenizer_checkpoint,
        device=cfg.device,
    )
    logger.info(
        f"  Tokenizer ready: n_q={tokenizer.n_q}, vocab_size={tokenizer.vocab_size}, "
        f"downsample_ratio={tokenizer.downsample_ratio}"
    )
    return tokenizer


def _extract_criss_cross_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "criss_cross_state_dict" in checkpoint:
        return checkpoint["criss_cross_state_dict"]
    raise KeyError(
        "Checkpoint does not contain 'state_dict' or 'criss_cross_state_dict'"
    )


def _extract_hparams(
    checkpoint: Dict[str, Any],
    architecture_checkpoint_path: Optional[str],
    device: str,
) -> Dict[str, Any]:
    if "hyper_parameters" in checkpoint:
        return dict(checkpoint["hyper_parameters"])
    if architecture_checkpoint_path is None:
        raise KeyError(
            "Promoted word-classification checkpoint lacks Lightning hyper_parameters; "
            "provide model.criss_cross_checkpoint as architecture source."
        )
    arch_checkpoint = torch.load(architecture_checkpoint_path, map_location=device)
    if "hyper_parameters" not in arch_checkpoint:
        raise KeyError(
            f"Architecture checkpoint {architecture_checkpoint_path} lacks hyper_parameters"
        )
    return dict(arch_checkpoint["hyper_parameters"])


def _write_checkpoint_load_report(
    report_path: Path,
    report: Dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"  Checkpoint load report written to {report_path}")


def _filter_state_dict_for_model(
    model: CrissCrossTransformerModule,
    source_state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    target_state_dict = model.state_dict()
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    loaded_keys = []
    skipped_keys = []
    mismatched_keys = []

    for key, value in source_state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]

        if "rope_embedding_layer.rotate" in key:
            skipped_keys.append({"key": key, "reason": "deterministic_rope_buffer"})
            continue

        if key not in target_state_dict:
            skipped_keys.append({"key": key, "reason": "not_in_target_model"})
            continue

        target_value = target_state_dict[key]
        if tuple(value.shape) == tuple(target_value.shape):
            filtered_state_dict[key] = value
            loaded_keys.append(key)
            continue

        if key == "sensor_type_layer.weight" and value.ndim == target_value.ndim and value.shape[1] == target_value.shape[1]:
            resized = target_value.clone()
            rows_to_copy = min(value.shape[0], target_value.shape[0])
            resized[:rows_to_copy] = value[:rows_to_copy].to(resized.device)
            filtered_state_dict[key] = resized
            loaded_keys.append(key)
            mismatched_keys.append({
                "key": key,
                "source_shape": list(value.shape),
                "target_shape": list(target_value.shape),
                "resolution": f"resized_rows_copied_{rows_to_copy}",
            })
            continue

        mismatched_keys.append({
            "key": key,
            "source_shape": list(value.shape),
            "target_shape": list(target_value.shape),
            "resolution": "skipped",
        })

    report = {
        "loaded_keys": loaded_keys,
        "skipped_keys": skipped_keys,
        "mismatched_keys": mismatched_keys,
    }
    return filtered_state_dict, report


def load_criss_cross_model(
    checkpoint_path: str,
    tokenizer: NeuroTokenizerAdapter,
    device: str = "cuda",
    num_sensor_types: Optional[int] = None,
    architecture_checkpoint_path: Optional[str] = None,
    report_path: Optional[Path] = None,
) -> CrissCrossTransformerModule:
    """
    Load CrissCrossTransformer from checkpoint.

    Args:
        checkpoint_path: Path to CrissCross checkpoint
        tokenizer: Loaded neural tokenizer adapter
        device: Device to load model on

    Returns:
        Loaded CrissCrossTransformerModule
    """
    logger.info(f"Loading CrissCross model from: {checkpoint_path}")

    # Load checkpoint manually to handle RoPE size mismatch
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    hparams = _extract_hparams(checkpoint, architecture_checkpoint_path, device)
    if num_sensor_types is not None:
        hparams['num_sensor_types'] = num_sensor_types
    hparams['vocab_size'] = int(tokenizer.vocab_size)

    # Create model instance with saved hyperparameters
    model = CrissCrossTransformerModule(
        tokenizer=tokenizer,
        **hparams
    )

    state_dict = _extract_criss_cross_state_dict(checkpoint)
    filtered_state_dict, report = _filter_state_dict_for_model(model, state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    report.update({
        "checkpoint_path": str(checkpoint_path),
        "architecture_checkpoint_path": str(architecture_checkpoint_path or checkpoint_path),
        "missing_keys_after_load": list(missing_keys),
        "unexpected_keys_after_load": list(unexpected_keys),
    })
    if report_path is not None:
        _write_checkpoint_load_report(report_path, report)

    logger.info(f"  Successfully loaded checkpoint (RoPE will auto-expand to 625 on first forward pass)")

    model.to(device)
    model.eval()  # Start in eval mode

    logger.info(f"  Latent dim: {model.latent_dim}")
    logger.info(f"  Num layers: {model.hparams.num_layers}")
    logger.info(f"  Num heads: {model.hparams.num_heads}")
    logger.info(f"  Loaded with strict=False (RoPE will auto-expand for sequence length)")

    return model


def initialize_criss_cross_from_scratch(
    checkpoint_path: str,
    tokenizer: NeuroTokenizerAdapter,
    device: str = "cuda",
    num_sensor_types: Optional[int] = None,
    report_path: Optional[Path] = None,
) -> CrissCrossTransformerModule:
    """
    Initialize CrissCross with random weights using architecture from checkpoint.

    This extracts hyperparameters from a pretrained checkpoint but creates
    a fresh model with randomly initialized weights instead of loading the
    trained parameters.

    Args:
        checkpoint_path: Path to checkpoint (used only for architecture params)
        tokenizer: Loaded neural tokenizer adapter
        device: Device to load model on

    Returns:
        Randomly initialized CrissCrossTransformerModule
    """
    logger.info(f"Initializing CrissCross from scratch using architecture from: {checkpoint_path}")

    # Load checkpoint to extract hyperparameters
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = dict(checkpoint['hyper_parameters'])
    if num_sensor_types is not None:
        hparams['num_sensor_types'] = num_sensor_types
    hparams['vocab_size'] = int(tokenizer.vocab_size)

    # Create model instance with saved hyperparameters but random weights
    model = CrissCrossTransformerModule(
        tokenizer=tokenizer,
        **hparams
    )

    logger.info(f"  Created model with random initialization")
    logger.info(f"  Latent dim: {model.latent_dim}")
    logger.info(f"  Num layers: {model.hparams.num_layers}")
    logger.info(f"  Num heads: {model.hparams.num_heads}")

    model.to(device)
    model.eval()  # Start in eval mode
    if report_path is not None:
        _write_checkpoint_load_report(report_path, {
            "mode": "train_from_scratch",
            "architecture_checkpoint_path": str(checkpoint_path),
            "loaded_keys": [],
            "skipped_keys": [],
            "mismatched_keys": [],
        })

    return model


def resolve_initial_checkpoint(cfg: DictConfig) -> Tuple[str, Optional[str], str]:
    architecture_checkpoint = cfg.model.get("criss_cross_checkpoint", None)
    if cfg.model.get("train_from_scratch", False):
        if not architecture_checkpoint or not Path(architecture_checkpoint).exists():
            raise FileNotFoundError(
                f"model.train_from_scratch=true requires an existing "
                f"model.criss_cross_checkpoint architecture source, got {architecture_checkpoint}"
            )
        return architecture_checkpoint, None, "scratch"

    if cfg.model.get("use_promoted_checkpoint", False):
        promoted = cfg.model.get("promoted_checkpoint", None)
        if not promoted or not Path(promoted).exists():
            raise FileNotFoundError(
                f"model.use_promoted_checkpoint=true but promoted checkpoint is missing: {promoted}"
            )
        if not architecture_checkpoint or not Path(architecture_checkpoint).exists():
            raise FileNotFoundError(
                "Promoted checkpoints require model.criss_cross_checkpoint "
                "as architecture source."
            )
        return promoted, architecture_checkpoint, "promoted"

    if not architecture_checkpoint or not Path(architecture_checkpoint).exists():
        raise FileNotFoundError(
            f"model.criss_cross_checkpoint not found: {architecture_checkpoint}"
        )
    return architecture_checkpoint, None, "pretrained"


def maybe_load_word_mlp_from_checkpoint(
    word_mlp: CrissCrossWordEmbeddingExtractor,
    checkpoint_path: str,
    report_path: Path,
    device: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "word_mlp_state_dict" not in checkpoint:
        return
    source_state = checkpoint["word_mlp_state_dict"]
    target_state = word_mlp.state_dict()
    filtered = {}
    loaded = []
    mismatched = []
    skipped = []
    for key, value in source_state.items():
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
            filtered[key] = value
            loaded.append(key)
        elif key in target_state:
            mismatched.append({
                "key": key,
                "source_shape": list(value.shape),
                "target_shape": list(target_state[key].shape),
                "resolution": "skipped",
            })
        else:
            skipped.append({"key": key, "reason": "not_in_target_word_mlp"})
    missing, unexpected = word_mlp.load_state_dict(filtered, strict=False)
    _write_checkpoint_load_report(report_path, {
        "checkpoint_path": str(checkpoint_path),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "mismatched_keys": mismatched,
        "missing_keys_after_load": list(missing),
        "unexpected_keys_after_load": list(unexpected),
    })


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_top_k_accuracy(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute top-k retrieval accuracy.

    For each predicted embedding, find the k most similar vocabulary embeddings
    and check if the true label is among them.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        k_values: List of k values to compute accuracy for

    Returns:
        metrics: Dict with top-k accuracy for each k
    """
    # Compute cosine similarity
    pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
    vocab_norm = F.normalize(vocab_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, vocab_norm.T)  # [N, vocab_size]

    metrics = {}
    for k in k_values:
        _, top_k_indices = torch.topk(similarity, k=k, dim=1)  # [N, k]

        # Check if true label is in top-k
        true_labels_expanded = true_labels.unsqueeze(1).expand(-1, k)
        hits = (top_k_indices == true_labels_expanded).any(dim=1)

        accuracy = hits.float().mean().item()
        metrics[f'top{k}_accuracy'] = accuracy

    return metrics


def compute_top_k_accuracy_with_retrieval_set(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    retrieval_set_size: int,
    k: int = 10
) -> Dict[str, Any]:
    """
    Compute top-k retrieval accuracy using a subset of most frequent words as retrieval set.

    Only samples whose true label is within the top retrieval_set_size words are evaluated.
    Retrieval is performed against only the retrieval set embeddings.

    Since vocabulary is ordered by frequency (most frequent first), the retrieval set
    consists of vocab indices 0 to retrieval_set_size-1.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices (0 = most frequent word)
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        retrieval_set_size: Number of most frequent words to use as retrieval set (e.g., 50, 250)
        k: K value for top-k retrieval (default: 10)

    Returns:
        metrics: Dict with:
            - topk_accuracy: Top-k accuracy within the retrieval set
            - n_samples: Number of samples evaluated (with labels in retrieval set)
            - n_skipped: Number of samples skipped (labels outside retrieval set)
    """
    # Filter to samples whose true label is in the retrieval set
    # Since vocab is ordered by frequency, indices 0 to retrieval_set_size-1 are most frequent
    in_retrieval_set = true_labels < retrieval_set_size
    n_samples = in_retrieval_set.sum().item()
    n_skipped = len(true_labels) - n_samples

    if n_samples == 0:
        return {
            f'top{k}_accuracy_retrieval{retrieval_set_size}': 0.0,
            f'n_samples_retrieval{retrieval_set_size}': 0,
            f'n_skipped_retrieval{retrieval_set_size}': n_skipped
        }

    # Get filtered predictions and labels
    filtered_pred = pred_embeddings[in_retrieval_set]  # [n_samples, 1024]
    filtered_labels = true_labels[in_retrieval_set]  # [n_samples]

    # Get retrieval set embeddings (top retrieval_set_size words)
    retrieval_embeddings = vocab_embeddings[:retrieval_set_size]  # [retrieval_set_size, 1024]

    # Compute cosine similarity against retrieval set only
    pred_norm = F.normalize(filtered_pred, p=2, dim=1)
    retrieval_norm = F.normalize(retrieval_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, retrieval_norm.T)  # [n_samples, retrieval_set_size]

    # Get top-k predictions (indices are now 0 to retrieval_set_size-1)
    actual_k = min(k, retrieval_set_size)
    _, top_k_indices = torch.topk(similarity, k=actual_k, dim=1)  # [n_samples, k]

    # Check if true label is in top-k
    filtered_labels_expanded = filtered_labels.unsqueeze(1).expand(-1, actual_k)
    hits = (top_k_indices == filtered_labels_expanded).any(dim=1)

    accuracy = hits.float().mean().item()

    return {
        f'top{k}_accuracy_retrieval{retrieval_set_size}': accuracy,
        f'n_samples_retrieval{retrieval_set_size}': n_samples,
        f'n_skipped_retrieval{retrieval_set_size}': n_skipped
    }


def compute_balanced_top_k_accuracy_with_retrieval_set(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    retrieval_set_size: int,
    k: int = 10
) -> float:
    """
    Compute balanced (macro-averaged) top-k retrieval accuracy using a retrieval subset.

    Only considers classes within the retrieval set. For each class in the retrieval set,
    compute top-k accuracy and then macro-average across classes.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        retrieval_set_size: Number of most frequent words to use as retrieval set
        k: K value for top-k retrieval

    Returns:
        balanced_accuracy: Macro-averaged top-k accuracy across retrieval set classes
    """
    # Filter to samples whose true label is in the retrieval set
    in_retrieval_set = true_labels < retrieval_set_size
    n_samples = in_retrieval_set.sum().item()

    if n_samples == 0:
        return 0.0

    # Get filtered predictions and labels
    filtered_pred = pred_embeddings[in_retrieval_set]
    filtered_labels = true_labels[in_retrieval_set]

    # Get retrieval set embeddings
    retrieval_embeddings = vocab_embeddings[:retrieval_set_size]

    # Compute cosine similarity against retrieval set only
    pred_norm = F.normalize(filtered_pred, p=2, dim=1)
    retrieval_norm = F.normalize(retrieval_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, retrieval_norm.T)

    actual_k = min(k, retrieval_set_size)
    _, top_k_indices = torch.topk(similarity, k=actual_k, dim=1)

    # Compute per-class accuracy for classes in retrieval set
    per_class_accuracies = []

    for class_idx in range(retrieval_set_size):
        class_mask = (filtered_labels == class_idx)
        n_class_samples = class_mask.sum().item()

        if n_class_samples == 0:
            continue

        class_top_k = top_k_indices[class_mask]
        hits = (class_top_k == class_idx).any(dim=1)
        class_acc = hits.float().mean().item()

        per_class_accuracies.append(class_acc)

    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return balanced_accuracy


def compute_balanced_top_k_accuracy(
    pred_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    k: int = 10
) -> float:
    """
    Compute balanced (macro-averaged) top-k retrieval accuracy.

    For each word class in the vocabulary, compute top-k accuracy and then
    macro-average across all classes. This gives equal weight to each class
    regardless of frequency.

    Args:
        pred_embeddings: [N, 1024] predicted word embeddings
        true_labels: [N] ground truth vocabulary indices
        vocab_embeddings: [vocab_size, 1024] T5 embeddings for all vocabulary words
        k: K value for top-k retrieval

    Returns:
        balanced_accuracy: Macro-averaged top-k accuracy across all vocabulary words
    """
    # Compute cosine similarity
    pred_norm = F.normalize(pred_embeddings, p=2, dim=1)
    vocab_norm = F.normalize(vocab_embeddings, p=2, dim=1)
    similarity = torch.matmul(pred_norm, vocab_norm.T)  # [N, vocab_size]

    _, top_k_indices = torch.topk(similarity, k=k, dim=1)  # [N, k]

    # Compute per-class accuracy
    vocab_size = vocab_embeddings.shape[0]
    per_class_accuracies = []

    for class_idx in range(vocab_size):
        # Get samples for this class
        class_mask = (true_labels == class_idx)
        n_samples = class_mask.sum().item()

        if n_samples == 0:
            # No samples for this class, skip
            continue

        # Check if true label is in top-k for this class
        class_top_k = top_k_indices[class_mask]  # [n_samples, k]
        hits = (class_top_k == class_idx).any(dim=1)
        class_acc = hits.float().mean().item()

        per_class_accuracies.append(class_acc)

    # Macro-average across classes
    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return balanced_accuracy


def compute_embedding_metrics(
    pred_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Compute embedding quality metrics.

    Args:
        pred_embeddings: [N, 1024] predicted embeddings
        target_embeddings: [N, 1024] target T5 embeddings

    Returns:
        metrics: Dict with cosine similarity and norm statistics
    """
    # Cosine similarity with target
    cos_sim = F.cosine_similarity(pred_embeddings, target_embeddings, dim=1)

    # Embedding norms
    pred_norms = torch.norm(pred_embeddings, p=2, dim=1)
    target_norms = torch.norm(target_embeddings, p=2, dim=1)

    return {
        'mean_cosine_similarity': cos_sim.mean().item(),
        'std_cosine_similarity': cos_sim.std().item(),
        'mean_pred_norm': pred_norms.mean().item(),
        'std_pred_norm': pred_norms.std().item(),
        'mean_target_norm': target_norms.mean().item(),
    }


# ============================================================================
# Training and Evaluation
# ============================================================================

def training_step(
    batch: Dict[str, Any],
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
    downsample_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training step handling 10 words per 30s sample.

    Args:
        batch: Dictionary from word_level_collate_fn
        criss_cross_model: CrissCross transformer
        word_mlp: Word embedding MLP
        vocab_embeddings: [vocab_size, 1024] T5 embeddings
        criterion: SigLIP loss function
        device: Device to run on

    Returns:
        loss: Scalar loss value
        word_embeddings: [B*10, 1024] predicted embeddings
        target_embeddings: [B*10, 1024] target embeddings
    """
    meg = batch['meg'].to(device)  # [B, C, 7500]
    word_labels = batch['word_labels'].to(device)  # [B*10]
    subsegment_info = batch['subsegment_info']
    sensor_xyzdir = batch['sensor_xyzdir'].to(device)
    sensor_xyz = sensor_xyzdir[..., :3]
    sensor_abc = sensor_xyzdir[..., 3:]
    sensor_types = batch['sensor_types'].to(device)
    sensor_mask = batch['sensor_mask'].to(device)

    # 1. Forward pass through CrissCross (no masking for evaluation)
    output = criss_cross_model(
        meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask,
        apply_mask=False
    )
    features = output['features']  # [B, C, 625, 512]

    # 2. Extract word embeddings for all subsegments
    word_embeddings = []

    for info in subsegment_info:
        b_idx = info['batch_idx']
        start_sample = info['start_sample']
        end_sample = info['end_sample']

        # Map to encoded timesteps
        start_t, end_t = map_raw_to_encoded_timesteps(start_sample, end_sample, downsample_ratio)

        # Extract features for this word
        word_features = features[b_idx, :, start_t:end_t, :]  # [C, T_subseg, 512]

        # Pass through word MLP
        word_emb = word_mlp(word_features)  # [1024]
        word_embeddings.append(word_emb)

    word_embeddings = torch.stack(word_embeddings)  # [B*10, 1024]

    # 3. Get target embeddings
    # Index on CPU, then move to device
    target_embeddings = vocab_embeddings[word_labels.cpu()].to(device)  # [B*10, 1024]

    # 4. Compute SigLIP loss
    loss = criterion(word_embeddings, target_embeddings, reweigh_positives=True)

    return loss, word_embeddings, target_embeddings


def evaluate_epoch(
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    dataloader: DataLoader,
    vocab_embeddings: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
    retrieval_set_sizes: List[int] = [50, 250],
    k: int = 10,
    downsample_ratio: int = 12,
) -> Dict[str, float]:
    """
    Evaluate on validation or test set.

    Computes top-k retrieval accuracy for each retrieval set size.
    For each retrieval set size, only samples with labels in that set are evaluated,
    and retrieval is performed against those embeddings.

    Args:
        criss_cross_model: CrissCross transformer
        word_mlp: Word embedding MLP
        dataloader: DataLoader for evaluation
        vocab_embeddings: [vocab_size, 1024] T5 embeddings (ordered by frequency)
        criterion: SigLIP loss function
        device: Device to run on
        retrieval_set_sizes: List of retrieval set sizes to evaluate (e.g., [50, 250])
        k: K value for top-k accuracy (default: 10)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    criss_cross_model.eval()
    word_mlp.eval()

    all_losses = []
    all_pred_embeddings = []
    all_target_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Skip batches with no valid words
            if len(batch['word_labels']) == 0:
                continue

            loss, pred_embs, target_embs = training_step(
                batch, criss_cross_model, word_mlp,
                vocab_embeddings, criterion, device, downsample_ratio
            )

            all_losses.append(loss.item())
            all_pred_embeddings.append(pred_embs.cpu())
            all_target_embeddings.append(target_embs.cpu())
            all_labels.append(batch['word_labels'])

    # Aggregate results
    all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)
    all_target_embeddings = torch.cat(all_target_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = {}
    metrics['loss'] = sum(all_losses) / len(all_losses)

    # Compute top-k accuracy for each retrieval set size
    for retrieval_size in retrieval_set_sizes:
        # Top-k accuracy with retrieval set
        retrieval_metrics = compute_top_k_accuracy_with_retrieval_set(
            all_pred_embeddings, all_labels, vocab_embeddings,
            retrieval_set_size=retrieval_size, k=k
        )
        metrics.update(retrieval_metrics)

        # Balanced top-k accuracy with retrieval set
        balanced_acc = compute_balanced_top_k_accuracy_with_retrieval_set(
            all_pred_embeddings, all_labels, vocab_embeddings,
            retrieval_set_size=retrieval_size, k=k
        )
        metrics[f'balanced_top{k}_accuracy_retrieval{retrieval_size}'] = balanced_acc

    # Embedding quality (computed on all samples)
    emb_metrics = compute_embedding_metrics(all_pred_embeddings, all_target_embeddings)
    metrics.update(emb_metrics)

    return metrics


def train_and_evaluate(
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: CrissCrossWordEmbeddingExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab_embeddings: torch.Tensor,
    cfg: DictConfig,
    device: str,
    downsample_ratio: int,
) -> Dict[str, float]:
    """
    Main training and evaluation loop.

    Args:
        criss_cross_model: CrissCross transformer
        word_mlp: Word embedding MLP
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        vocab_embeddings: [vocab_size, 1024] T5 embeddings
        cfg: Hydra configuration
        device: Device to run on

    Returns:
        test_metrics: Final test set metrics
    """
    # Setup optimizer with mode-appropriate learning rates
    if cfg.model.train_from_scratch:
        # From scratch: use same LR for both components
        logger.info(f"Optimizer: Using from_scratch_lr={cfg.training.from_scratch_lr} for all parameters")
        params = [
            {'params': criss_cross_model.parameters(), 'lr': cfg.training.from_scratch_lr},
            {'params': word_mlp.parameters(), 'lr': cfg.training.from_scratch_lr}
        ]
    else:
        # Fine-tuning: use differential learning rates
        logger.info(f"Optimizer: criss_cross_lr={cfg.training.criss_cross_lr}, word_mlp_lr={cfg.training.word_mlp_lr}")
        params = [
            {'params': criss_cross_model.parameters(), 'lr': cfg.training.criss_cross_lr},
            {'params': word_mlp.parameters(), 'lr': cfg.training.word_mlp_lr}
        ]

    optimizer = AdamW(params, weight_decay=cfg.training.weight_decay)

    # Setup scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    # Setup loss
    criterion = SigLipLoss(
        norm_kind=cfg.loss.norm_kind,
        temperature=cfg.loss.temperature,
        bias=cfg.loss.bias,
        reduction=cfg.loss.reduction
    ).to(device)

    # Training loop
    best_val_top10_acc = 0.0
    patience_counter = 0
    best_test_metrics_at_best_val = {}  # Track test metrics at best validation
    best_val_epoch = 0  # Track which epoch had best validation
    start_epoch = 0
    previous_val_primary_acc = None
    checkpoint_dir = resolve_checkpoint_dir(cfg.logging)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified
    resume_checkpoint = cfg.training.get('resume_checkpoint', None)
    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info(f"\nResuming from checkpoint: {resume_checkpoint}")
        resume_ckpt = torch.load(resume_checkpoint, map_location=device)

        # Load model states
        criss_cross_model.load_state_dict(resume_ckpt['criss_cross_state_dict'])
        word_mlp.load_state_dict(resume_ckpt['word_mlp_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])

        # Load scheduler state
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])

        # Restore training state
        start_epoch = resume_ckpt.get('epoch', 0) + 1
        best_val_top10_acc = resume_ckpt.get('best_val_top10_acc', 0.0)
        patience_counter = resume_ckpt.get('patience_counter', 0)
        best_val_epoch = resume_ckpt.get('best_val_epoch', 0)
        best_test_metrics_at_best_val = resume_ckpt.get('best_test_metrics_at_best_val', {})

        logger.info(f"  Resumed from epoch {start_epoch}")
        logger.info(f"  Best val acc so far: {best_val_top10_acc:.4f} (epoch {best_val_epoch})")
        logger.info(f"  Patience counter: {patience_counter}")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")

        # Training
        criss_cross_model.train()
        criss_cross_model.enable_gradient_checkpointing()
        word_mlp.train()

        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Skip batches with no valid words
            if len(batch['word_labels']) == 0:
                continue

            optimizer.zero_grad()

            loss, pred_embs, target_embs = training_step(
                batch, criss_cross_model, word_mlp,
                vocab_embeddings, criterion, device, downsample_ratio
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(criss_cross_model.parameters()) + list(word_mlp.parameters()),
                cfg.training.gradient_clip_val
            )

            optimizer.step()

            train_losses.append(loss.item())

            # Log every N steps
            if batch_idx % cfg.logging.log_every_n_steps == 0:
                wandb.log({
                    'train/loss_step': loss.item(),
                    'train/step': epoch * len(train_loader) + batch_idx
                })

        # Compute epoch metrics
        train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"  Train loss: {train_loss:.4f}")

        # Validation
        val_metrics = evaluate_epoch(
            criss_cross_model, word_mlp, val_loader,
            vocab_embeddings, criterion, device,
            retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
            k=cfg.evaluation.k,
            downsample_ratio=downsample_ratio,
        )

        # Also evaluate on test set to understand dynamics (but don't use for early stopping)
        test_metrics = evaluate_epoch(
            criss_cross_model, word_mlp, test_loader,
            vocab_embeddings, criterion, device,
            retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
            k=cfg.evaluation.k,
            downsample_ratio=downsample_ratio,
        )

        # Get primary retrieval set size for early stopping (largest in list)
        primary_retrieval_size = cfg.evaluation.retrieval_set_sizes[-1]
        k = cfg.evaluation.k
        primary_metric_key = f'balanced_top{k}_accuracy_retrieval{primary_retrieval_size}'
        val_primary_acc = val_metrics.get(primary_metric_key, 0)
        primary_delta = (
            None if previous_val_primary_acc is None
            else val_primary_acc - previous_val_primary_acc
        )
        previous_best_val_acc = best_val_top10_acc
        primary_margin_over_best = val_primary_acc - previous_best_val_acc
        improved_this_epoch = val_primary_acc > previous_best_val_acc + cfg.training.min_delta
        primary_gain_over_best = primary_margin_over_best if improved_this_epoch else 0.0

        logger.info(f"  Val loss: {val_metrics['loss']:.4f}")
        for ret_size in cfg.evaluation.retrieval_set_sizes:
            val_acc = val_metrics.get(f'top{k}_accuracy_retrieval{ret_size}', 0)
            val_balanced = val_metrics.get(f'balanced_top{k}_accuracy_retrieval{ret_size}', 0)
            val_n = val_metrics.get(f'n_samples_retrieval{ret_size}', 0)
            test_acc = test_metrics.get(f'top{k}_accuracy_retrieval{ret_size}', 0)
            test_balanced = test_metrics.get(f'balanced_top{k}_accuracy_retrieval{ret_size}', 0)
            logger.info(f"  [Retrieval {ret_size}] Val top-{k}: {val_acc:.4f}, balanced: {val_balanced:.4f} (n={val_n})")
            logger.info(f"  [Retrieval {ret_size}] Test top-{k}: {test_acc:.4f}, balanced: {test_balanced:.4f}")

        # Log to WandB
        log_dict = {
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'val/primary_metric_value': val_primary_acc,
            'val/primary_metric_margin_over_previous_best': primary_margin_over_best,
            'val/primary_metric_gain_over_previous_best': primary_gain_over_best,
            'val/primary_metric_improved': int(improved_this_epoch),
            **{f'val/{metric_k}': v for metric_k, v in val_metrics.items()},
            **{f'test_during_train/{metric_k}': v for metric_k, v in test_metrics.items()}
        }
        if primary_delta is not None:
            log_dict['val/primary_metric_delta_from_previous_epoch'] = primary_delta

        # Log best test metrics at best validation (tracks test performance at best val checkpoint so far)
        if best_test_metrics_at_best_val:
            log_dict.update({f'test_at_best_val/{metric_k}': v for metric_k, v in best_test_metrics_at_best_val.items()})
            log_dict['test_at_best_val/best_val_epoch'] = best_val_epoch

        wandb.log(log_dict)

        # Use primary retrieval set's balanced accuracy for early stopping
        scheduler.step(val_primary_acc)

        # Early stopping and checkpointing
        if improved_this_epoch:
            best_val_top10_acc = val_primary_acc
            patience_counter = 0
            best_test_metrics_at_best_val = test_metrics.copy()  # Update best test metrics
            best_val_epoch = epoch + 1  # Track which epoch had best validation (1-indexed)

            # Save best model (include test_metrics for comparison)
            checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
            torch.save({
                'epoch': epoch,
                'criss_cross_state_dict': criss_cross_model.state_dict(),
                'word_mlp_state_dict': word_mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics_at_best_val': test_metrics,  # Test metrics at this epoch
                'best_val_top10_acc': best_val_top10_acc,
                'patience_counter': patience_counter,
                'best_val_epoch': best_val_epoch,
                'best_test_metrics_at_best_val': best_test_metrics_at_best_val,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, checkpoint_path)
            test_primary_acc = test_metrics.get(f'top{k}_accuracy_retrieval{primary_retrieval_size}', 0)
            logger.info(f"  Saved best model (val balanced: {best_val_top10_acc:.4f}, test top-{k}@{primary_retrieval_size}: {test_primary_acc:.4f})")
        else:
            patience_counter += 1

        # Save latest checkpoint for resuming (every epoch)
        latest_checkpoint_path = checkpoint_dir / 'checkpoint_latest.pt'
        torch.save({
            'epoch': epoch,
            'criss_cross_state_dict': criss_cross_model.state_dict(),
            'word_mlp_state_dict': word_mlp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'best_val_top10_acc': best_val_top10_acc,
            'patience_counter': patience_counter,
            'best_val_epoch': best_val_epoch,
            'best_test_metrics_at_best_val': best_test_metrics_at_best_val,
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, latest_checkpoint_path)

        history_row = {
            'epoch': epoch + 1,
            'primary_metric': primary_metric_key,
            'train/loss': train_loss,
            'val/primary_metric_value': val_primary_acc,
            'val/primary_metric_delta_from_previous_epoch': primary_delta,
            'val/primary_metric_margin_over_previous_best': primary_margin_over_best,
            'val/primary_metric_gain_over_previous_best': primary_gain_over_best,
            'val/primary_metric_improved': improved_this_epoch,
            'val/best_primary_metric': best_val_top10_acc,
            'val/best_epoch': best_val_epoch,
            'training/patience_counter': patience_counter,
            'training/early_stopped': patience_counter >= cfg.training.patience,
            **{f'optimizer/lr_group_{i}': group['lr'] for i, group in enumerate(optimizer.param_groups)},
            **{f'val/{metric_k}': v for metric_k, v in val_metrics.items()},
            **{f'test_during_train/{metric_k}': v for metric_k, v in test_metrics.items()},
            **{f'test_at_best_val/{metric_k}': v for metric_k, v in best_test_metrics_at_best_val.items()},
        }
        csv_path, jsonl_path = append_epoch_metrics_history(
            cfg.logging.save_dir,
            history_row,
            reset=(epoch == start_epoch and start_epoch == 0),
        )
        logger.info(f"  Metrics history updated: {csv_path} and {jsonl_path}")
        previous_val_primary_acc = val_primary_acc

        if patience_counter >= cfg.training.patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model and test
    logger.info("\nLoading best model for final evaluation...")
    checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    criss_cross_model.load_state_dict(checkpoint['criss_cross_state_dict'])
    word_mlp.load_state_dict(checkpoint['word_mlp_state_dict'])

    # Get test metrics from when checkpoint was saved (in-memory evaluation)
    test_metrics_at_best_val = checkpoint.get('test_metrics_at_best_val', {})

    # Evaluate again after loading (to detect save/load issues)
    test_metrics_after_load = evaluate_epoch(
        criss_cross_model, word_mlp, test_loader,
        vocab_embeddings, criterion, device,
        retrieval_set_sizes=cfg.evaluation.retrieval_set_sizes,
        k=cfg.evaluation.k,
        downsample_ratio=downsample_ratio,
    )

    logger.info("\n=== Final Test Results Comparison ===")
    logger.info("Test metrics at best val epoch (in-memory):")
    for metric_k, v in test_metrics_at_best_val.items():
        logger.info(f"  {metric_k}: {v:.4f}")

    logger.info("\nTest metrics after checkpoint load:")
    for metric_k, v in test_metrics_after_load.items():
        logger.info(f"  {metric_k}: {v:.4f}")

    # Check for discrepancy using primary retrieval set (largest)
    primary_retrieval_size = cfg.evaluation.retrieval_set_sizes[-1]
    eval_k = cfg.evaluation.k
    if test_metrics_at_best_val:
        primary_key = f'top{eval_k}_accuracy_retrieval{primary_retrieval_size}'
        in_mem_acc = test_metrics_at_best_val.get(primary_key, 0)
        loaded_acc = test_metrics_after_load.get(primary_key, 0)
        diff = abs(in_mem_acc - loaded_acc)
        if diff > 0.01:
            logger.warning(f"\n⚠️  Discrepancy detected! In-memory: {in_mem_acc:.4f}, After load: {loaded_acc:.4f}, Diff: {diff:.4f}")
        else:
            logger.info(f"\n✓ No significant discrepancy (diff: {diff:.4f})")

    wandb.log({
        **{f'test_in_memory/{k}': v for k, v in test_metrics_at_best_val.items()},
        **{f'test_after_load/{k}': v for k, v in test_metrics_after_load.items()}
    })

    return test_metrics_after_load


# ============================================================================
# Main Entry Point
# ============================================================================

@hydra.main(version_base=None, config_path="../configs", config_name="eval_criss_cross_word_classification")
def main(cfg: DictConfig):
    """Main entry point for CrissCross word classification evaluation."""

    # Setup output directory
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resolve_checkpoint_dir(cfg.logging)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _run_log_file = _install_run_tee(save_dir)

    # Setup logging after stdout/stderr tee so logs are persisted with console output.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    logger.info(f"Metrics/results directory: {save_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    config_snapshot = write_resolved_config_snapshot(save_dir, cfg)
    logger.info(f"Resolved config snapshot: {config_snapshot}")

    # Setup WandB
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config['training_mode'] = 'from_scratch' if cfg.model.train_from_scratch else 'fine_tuning'

    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.experiment_name + '-scratch' if cfg.model.train_from_scratch else cfg.logging.experiment_name,
        config=wandb_config
    )

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # 1. Load tokenizer
    tokenizer = load_tokenizer_from_config(cfg)

    # 2. Load or initialize CrissCross model
    num_sensor_types = get_num_sensor_types_for_config(cfg)
    logger.info(f"Using {num_sensor_types} sensor type embeddings")
    init_checkpoint, architecture_checkpoint, init_mode = resolve_initial_checkpoint(cfg)
    report_path = checkpoint_dir / "checkpoint_load_report.json"
    if init_mode == "scratch":
        logger.info("Training mode: FROM SCRATCH (random initialization)")
        criss_cross_model = initialize_criss_cross_from_scratch(
            init_checkpoint,
            tokenizer,
            cfg.device,
            num_sensor_types=num_sensor_types,
            report_path=report_path,
        )
    else:
        logger.info(f"Training mode: {init_mode.upper()} initialization")
        criss_cross_model = load_criss_cross_model(
            init_checkpoint,
            tokenizer,
            cfg.device,
            num_sensor_types=num_sensor_types,
            architecture_checkpoint_path=architecture_checkpoint,
            report_path=report_path,
        )

    # 3. Build vocabulary - will be done after dataset creation for both split modes
    # Note: Vocabulary now includes ALL words (no top-K filtering)
    # For hashed split, vocabulary will be built after dataset creation
    vocab = None
    word_to_idx = None
    vocab_embeddings = None

    # 5. Create datasets
    logger.info("\nCreating datasets...")

    # Get dataset metadata based on config
    dataset_type = cfg.data.get('dataset_type', 'armeni')
    max_channel_dim = resolve_max_channel_dim(cfg)

    logger.info(f"Using dataset type: {dataset_type}")
    if cfg.data.get("datasets") is not None:
        logger.info(f"Using pooled datasets: {[entry.get('dataset_name', entry.get('dataset_type')) for entry in cfg.data.datasets]}")
    logger.info(f"Max channel dim: {max_channel_dim}")

    if cfg.data.get('use_hashed_split', False):
        logger.info("Using hashed sentence-based split...")

        # Create single dataset with ALL sessions/dataset entries
        full_dataset = create_word_dataset(cfg, sessions=_as_list(cfg.data.get("all_sessions", None)))

        logger.info(f"Total segments across all sessions: {len(full_dataset)}")

        train_indices, val_indices, test_indices, idx_to_group = assign_no_leak_splits(full_dataset, cfg)
        sentence_counts = Counter(" ".join(get_dataset_words(full_dataset, idx)) for idx in range(len(full_dataset)))

        # Log statistics
        total_sentences = len(sentence_counts)
        duplicate_sentences = sum(1 for count in sentence_counts.values() if count > 1)
        logger.info(f"Unique sentences: {total_sentences}")
        logger.info(f"Sentences appearing multiple times: {duplicate_sentences}")
        logger.info(f"Split sizes before train subsampling:")
        logger.info(f"  Train: {len(train_indices)} segments ({len(train_indices)/len(full_dataset)*100:.1f}%)")
        logger.info(f"  Val: {len(val_indices)} segments ({len(val_indices)/len(full_dataset)*100:.1f}%)")
        logger.info(f"  Test: {len(test_indices)} segments ({len(test_indices)/len(full_dataset)*100:.1f}%)")

        # Create subset datasets
        train_dataset_full = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # Subsample training data to train_pct
        total_size = len(train_dataset_full)
        sample_size = int(cfg.data.train_pct * total_size)
        remaining_size = total_size - sample_size

        logger.info(f"\nSubsampling training data to {cfg.data.train_pct*100}%...")
        logger.info(f"  Original: {total_size} segments")
        logger.info(f"  Subsampled: {sample_size} segments")

        train_dataset, _ = random_split(
            train_dataset_full,
            [sample_size, remaining_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )
        split_sizes = {
            "train_full": int(total_size),
            "train": int(len(train_dataset)),
            "val": int(len(val_dataset)),
            "test": int(len(test_dataset)),
            "total": int(len(full_dataset)),
        }

        # Validate no sentence leakage across splits
        logger.info("\nValidating split integrity...")

        # Reuse cached sentences from hashing step (no need to reconstruct)
        train_sentences = {idx_to_group[idx] for idx in train_indices}
        val_sentences = {idx_to_group[idx] for idx in val_indices}
        test_sentences = {idx_to_group[idx] for idx in test_indices}

        # Check for overlaps
        train_val_overlap = train_sentences & val_sentences
        train_test_overlap = train_sentences & test_sentences
        val_test_overlap = val_sentences & test_sentences

        logger.info(f"Split validation results:")
        logger.info(f"  Unique train sentences: {len(train_sentences)}")
        logger.info(f"  Unique val sentences: {len(val_sentences)}")
        logger.info(f"  Unique test sentences: {len(test_sentences)}")
        logger.info(f"  Train-Val overlap: {len(train_val_overlap)} (should be 0)")
        logger.info(f"  Train-Test overlap: {len(train_test_overlap)} (should be 0)")
        logger.info(f"  Val-Test overlap: {len(val_test_overlap)} (should be 0)")

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError("Sentence leakage detected across splits!")

        # Build vocabulary from ENTIRE dataset using ALL unique words
        # Training uses all words (no OOV), evaluation filters by retrieval set
        logger.info("\nBuilding vocabulary from entire dataset (ALL words)...")
        vocab, word_counter = build_vocabulary_from_word_datasets([full_dataset])

        logger.info(f"  Total unique words (vocabulary size): {len(vocab)}")
        logger.info(f"  Most frequent word: '{vocab[0]}' ({word_counter[vocab[0]]} occurrences)")
        logger.info(f"  Least frequent word: '{vocab[-1]}' ({word_counter[vocab[-1]]} occurrences)")

        # Create word-to-index mapping
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Generate T5 embeddings
        vocab_embeddings = generate_word_embeddings(
            vocab,
            vocab_size=len(vocab),
            layer=cfg.t5.layer,
            cache_dir=cfg.t5.cache_dir,
            device=cfg.device,
            dataset_type=dataset_type
        )

    else:
        # Use existing session-based temporal split
        logger.info("Using session-based temporal split...")

        train_dataset = create_word_dataset(cfg, sessions=_as_list(cfg.data.train_sessions))

        print("Original training dataset size:", len(train_dataset))
        print("Sampling", cfg.data.train_pct * 100, "% of training data...")

        total_size = len(train_dataset)
        sample_size = int(cfg.data.train_pct * total_size)
        remaining_size = total_size - sample_size

        print("New training dataset size:", sample_size)

        train_subset, _ = random_split(
            train_dataset,
            [sample_size, remaining_size],
            generator=torch.Generator().manual_seed(cfg.seed)
        )

        train_dataset = train_subset

        val_dataset = create_word_dataset(cfg, sessions=_as_list(cfg.data.val_sessions))
        test_dataset = create_word_dataset(cfg, sessions=_as_list(cfg.data.test_sessions))
        split_sizes = {
            "train_full": int(total_size),
            "train": int(len(train_dataset)),
            "val": int(len(val_dataset)),
            "test": int(len(test_dataset)),
            "total": int(len(train_dataset) + len(val_dataset) + len(test_dataset)),
        }

        # Build vocabulary from ALL datasets using ALL unique words
        # Training uses all words (no OOV), evaluation filters by retrieval set
        logger.info("\nBuilding vocabulary from all datasets (ALL words)...")
        # Get underlying dataset from train_dataset (which is a Subset)
        train_base = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset

        vocab, word_counter = build_vocabulary_from_word_datasets([train_base, val_dataset, test_dataset])

        logger.info(f"  Total unique words (vocabulary size): {len(vocab)}")
        logger.info(f"  Most frequent word: '{vocab[0]}' ({word_counter[vocab[0]]} occurrences)")
        logger.info(f"  Least frequent word: '{vocab[-1]}' ({word_counter[vocab[-1]]} occurrences)")

        # Create word-to-index mapping
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Generate T5 embeddings
        vocab_embeddings = generate_word_embeddings(
            vocab,
            vocab_size=len(vocab),
            layer=cfg.t5.layer,
            cache_dir=cfg.t5.cache_dir,
            device=cfg.device,
            dataset_type=dataset_type
        )

    logger.info(f"\nFinal dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} segments")
    logger.info(f"  Val: {len(val_dataset)} segments")
    logger.info(f"  Test: {len(test_dataset)} segments")

    # Create collate function
    collate_fn = create_word_level_collate_fn(word_to_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=collate_fn
    )

    # 6. Create word embedding MLP
    # Use the max_channel_dim computed earlier (based on dataset_type or config override)
    num_channels = max_channel_dim

    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=num_channels,
        latent_dim=criss_cross_model.latent_dim,
        embed_dim=cfg.model.word_mlp.embed_dim,
        hidden_dim=cfg.model.word_mlp.hidden_dim,
        dropout=cfg.model.word_mlp.dropout
    ).to(cfg.device)

    logger.info(f"\nWord MLP:")
    logger.info(f"  Input dim: {num_channels * criss_cross_model.latent_dim}")
    logger.info(f"  Hidden dim: {cfg.model.word_mlp.hidden_dim}")
    logger.info(f"  Output dim: {cfg.model.word_mlp.embed_dim}")

    if init_mode == "promoted":
        maybe_load_word_mlp_from_checkpoint(
            word_mlp,
            init_checkpoint,
            checkpoint_dir / "word_mlp_checkpoint_load_report.json",
            cfg.device,
        )

    metadata_path = write_run_metadata(
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        init_mode=init_mode,
        init_checkpoint=init_checkpoint,
        architecture_checkpoint=architecture_checkpoint,
        split_sizes=split_sizes,
        channel_count=num_channels,
        tokenizer=tokenizer,
        config_snapshot=config_snapshot,
    )
    logger.info(f"Run metadata saved to: {metadata_path}")

    # 7. Train and evaluate
    test_metrics = train_and_evaluate(
        criss_cross_model, word_mlp,
        train_loader, val_loader, test_loader,
        vocab_embeddings, cfg, cfg.device,
        downsample_ratio=int(tokenizer.downsample_ratio),
    )

    # 8. Save final results
    results_path = save_dir / 'final_results.txt'
    with open(results_path, 'w') as f:
        f.write("CrissCross Word Classification - Final Test Results\n")
        f.write("=" * 60 + "\n\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    final_results_json = {
        "experiment_name": str(cfg.logging.get("experiment_name", save_dir.name)),
        "task_mode": str(cfg.data.get("task_mode", "")),
        "target_sfreq": float(cfg.data.get("target_sfreq", 0.0)),
        "tokenizer_name": str(cfg.model.get("tokenizer_name", "biocodec")),
        "training_mode": _training_mode(cfg, init_mode),
        "seed": int(cfg.seed),
        "test_metrics": test_metrics,
        "checkpoint_best": str(checkpoint_dir / "checkpoint_best.pt"),
        "checkpoint_latest": str(checkpoint_dir / "checkpoint_latest.pt"),
        "run_metadata": str(metadata_path),
        "timestamp": datetime_now_iso(),
    }
    _write_json(save_dir / "final_results.json", final_results_json)

    write_run_metadata(
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        init_mode=init_mode,
        init_checkpoint=init_checkpoint,
        architecture_checkpoint=architecture_checkpoint,
        split_sizes=split_sizes,
        channel_count=num_channels,
        tokenizer=tokenizer,
        config_snapshot=config_snapshot,
        extra={"final_metrics": test_metrics},
    )

    logger.info(f"\nResults saved to: {results_path}")

    wandb.finish()
    _run_log_file.flush()

    return test_metrics


if __name__ == "__main__":
    main()
