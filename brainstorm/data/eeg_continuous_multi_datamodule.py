"""Multi-dataset DataModule for continuous EEG pre-training.

This module has the same public class name and nearly the same constructor as
``eeg_multi_datamodule.py`` so the existing EEG training script can use it
without changing its training loop. Word-alignment arguments are accepted for
compatibility but intentionally ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from .eeg_multi_dataset import MultiEEGDataset
from .eeg_word_aligned_dataset import scan_bids_eeg_channel_counts
from .eegdash_eeg_continuous_dataset import (
    EEGDashEEGContinuousDataset,
    scan_eegdash_eeg_channel_counts,
)
from .openneuro_eeg_continuous_dataset import OpenNeuroEEGContinuousDataset
from .sparrkulee_eeg_continuous_dataset import (
    SparrKULeeEEGContinuousDataset,
    scan_sparrkulee_eeg_channel_counts,
)
from .subsampled_dataset import SubsampledRecordingDataset
from .zuco_eeg_continuous_dataset import (
    ZuCoEEGContinuousDataset,
    scan_zuco_eeg_channel_counts,
)


_DATASET_ALIASES = {
    "ds004408": "openneuro_ds004408",
    "openneuro_ds004408": "openneuro_ds004408",
    "openneuroeeg_ds004408": "openneuro_ds004408",
    "openneuroEEG_ds004408": "openneuro_ds004408",
    "ds007808": "openneuro_ds007808",
    "openneuro_ds007808": "openneuro_ds007808",
    "openneuroeeg_ds007808": "openneuro_ds007808",
    "openneuroEEG_ds007808": "openneuro_ds007808",
    "sparrkulee": "sparrkulee",
    "sparrkulee_eeg": "sparrkulee",
    "sparrkuleeeeg": "sparrkulee",
    "eegdash": "eegdash",
    "eegdash_nm000228": "eegdash",
    "nm000228": "eegdash",
    "zuco": "zuco",
    "zuco2": "zuco",
    "zuco_2": "zuco",
}

_DEFAULT_TASKS = {
    "openneuro_ds004408": ["listening"],
    "openneuro_ds007808": ["listening", "listeningcovert"],
    "sparrkulee": ["listeningActive"],
    "eegdash": ["delong", "control"],
    "zuco": ["NR"],
}


def _as_list(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    return [str(value) for value in values]


def _norm_id(value: str, prefix: str) -> str:
    text = str(value)
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _strip_prefix(value: str, prefix: str) -> str:
    text = str(value)
    return text[len(prefix) :] if text.startswith(prefix) else text


def _sorted_dirs(root: Path, pattern: str) -> List[str]:
    return sorted(path.name for path in root.glob(pattern) if path.is_dir())


def _resolve_eegdash_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    nested = root / "nm000228"
    return nested if nested.is_dir() else root


def _resolve_zuco_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if (root / "task1 - NR" / "Preprocessed").is_dir():
        return root
    nested = root / "data" / "zuco2"
    return nested if (nested / "task1 - NR" / "Preprocessed").is_dir() else root


def _discover_subjects(data_root: str | Path) -> List[str]:
    return _sorted_dirs(Path(data_root), "sub-*")


def _discover_zuco_subjects(data_root: str | Path) -> List[str]:
    preprocessed = _resolve_zuco_root(data_root) / "task1 - NR" / "Preprocessed"
    if not preprocessed.is_dir():
        return []
    return sorted(
        f"sub-{path.name}"
        for path in preprocessed.iterdir()
        if path.is_dir()
    )


def _discover_sessions(
    data_root: str | Path,
    subjects: Optional[Sequence[str]] = None,
) -> List[str]:
    root = Path(data_root)
    subject_names = list(subjects) if subjects is not None else _discover_subjects(root)
    sessions = set()
    for subject in subject_names:
        for session_dir in (root / _norm_id(subject, "sub-")).glob("ses-*"):
            if session_dir.is_dir():
                sessions.add(session_dir.name)
    return sorted(sessions)


def _exclude(
    values: Optional[Sequence[str]],
    excluded: Sequence[str],
    prefix: str,
) -> Optional[List[str]]:
    if values is None:
        return None
    excluded_norm = {
        _strip_prefix(_norm_id(item, prefix), prefix).lower() for item in excluded
    }
    kept = []
    for value in values:
        full = _norm_id(value, prefix)
        if _strip_prefix(full, prefix).lower() not in excluded_norm:
            kept.append(full)
    return kept


def _fraction_holdout(
    values: Sequence[str],
    fraction: float,
    seed: int,
) -> List[str]:
    unique = sorted(set(str(value) for value in values))
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {fraction}")
    if len(unique) < 2:
        raise ValueError(
            "val_fraction requires at least two discoverable split groups; "
            f"found {len(unique)}"
        )

    n_val = int(round(len(unique) * fraction))
    n_val = min(len(unique) - 1, max(1, n_val))
    rng = np.random.RandomState(int(seed))
    selected = rng.choice(len(unique), size=n_val, replace=False)
    return sorted(unique[int(index)] for index in selected.tolist())


class MultiEEGDataModule(pl.LightningDataModule):
    """Combine multiple continuous EEG datasets for pre-training.

    ``words_per_segment``, ``window_onset_offset`` and
    ``allow_missing_word_alignment`` remain in the signature only because the
    current training entrypoint passes them. They do not affect continuous
    pre-training.
    """

    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache/eeg_continuous",
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        use_recording_sampler: bool = True,
        sampler_seed: int = 42,
        debug_mode: bool = False,
        max_channel_dim: Optional[int] = None,
        infer_max_channel_dim: bool = True,
        recording_subsample_prop: Optional[float] = None,
        allow_missing_word_alignment: bool = False,
        tokenizer_name: str = "biocodec",
        **_: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.datasets_config = datasets_config
        self.segment_length = float(segment_length)
        self.subsegment_duration = float(subsegment_duration)
        self.cache_dir = str(cache_dir)
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.target_sfreq = float(target_sfreq)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.use_recording_sampler = bool(use_recording_sampler)
        self.sampler_seed = int(sampler_seed)
        self.debug_mode = bool(debug_mode)
        self.max_channel_dim = max_channel_dim
        self.infer_max_channel_dim = bool(infer_max_channel_dim)
        self.recording_subsample_prop = recording_subsample_prop
        self.tokenizer_name = str(tokenizer_name)
        self._split_cache: Dict[Tuple[Any, ...], List[str]] = {}

        # Accepted only for compatibility with the old word-aligned entrypoint.
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.allow_missing_word_alignment = allow_missing_word_alignment

        if recording_subsample_prop is not None and not (
            0.0 < recording_subsample_prop <= 1.0
        ):
            raise ValueError(
                "recording_subsample_prop must be in (0.0, 1.0], got "
                f"{recording_subsample_prop}"
            )

        self.train_dataset = None
        self.val_dataset = None

    @staticmethod
    def _canonical_type(dataset_type: str) -> str:
        key = str(dataset_type)
        return _DATASET_ALIASES.get(key, _DATASET_ALIASES.get(key.lower(), key))

    def _default_tasks(self, dataset_type: str) -> List[str]:
        canonical = self._canonical_type(dataset_type)
        if canonical not in _DEFAULT_TASKS:
            raise ValueError(
                f"Unknown EEG dataset type: {dataset_type}. Expected one of "
                f"{sorted(_DEFAULT_TASKS)}"
            )
        return list(_DEFAULT_TASKS[canonical])

    def _dataset_subjects(
        self,
        canonical: str,
        data_root: str,
    ) -> List[str]:
        if canonical == "eegdash":
            return _discover_subjects(_resolve_eegdash_root(data_root))
        if canonical == "zuco":
            return _discover_zuco_subjects(data_root)
        return _discover_subjects(data_root)

    def _dataset_sessions(
        self,
        canonical: str,
        data_root: str,
        subjects: Optional[Sequence[str]],
    ) -> List[str]:
        if canonical == "zuco":
            raise ValueError("ZuCo validation must be split by subject, not session")
        root = _resolve_eegdash_root(data_root) if canonical == "eegdash" else Path(data_root)
        return _discover_sessions(root, subjects=subjects)

    def _infer_max_channel_dim(self) -> Optional[int]:
        counts = []
        for config in self.datasets_config:
            canonical = self._canonical_type(config["type"])
            tasks = config.get("tasks", self._default_tasks(canonical))
            if canonical == "sparrkulee":
                counts.extend(
                    scan_sparrkulee_eeg_channel_counts(
                        config["data_root"],
                        tasks=tasks,
                    )
                )
            elif canonical == "eegdash":
                counts.extend(
                    scan_eegdash_eeg_channel_counts(
                        config["data_root"],
                        tasks=tasks,
                    )
                )
            elif canonical == "zuco":
                counts.extend(
                    scan_zuco_eeg_channel_counts(
                        config["data_root"],
                        tasks=tasks,
                    )
                )
            else:
                counts.extend(
                    scan_bids_eeg_channel_counts(config["data_root"], tasks=tasks)
                )
        if not counts:
            return None
        return max(item.n_channels for item in counts)

    def _resolve_max_channel_dim(self) -> Optional[int]:
        if self.max_channel_dim is not None:
            return int(self.max_channel_dim)
        if not self.infer_max_channel_dim:
            return None

        inferred = self._infer_max_channel_dim()
        if inferred is None:
            print(
                "Could not infer max_channel_dim from EEG headers/channels.tsv. "
                "Set data.max_channel_dim manually if channel counts differ."
            )
        else:
            print(f"Inferred continuous EEG max_channel_dim: {inferred}")
        return inferred

    def _fraction_split_groups(
        self,
        config: Dict[str, Any],
        subjects: Optional[List[str]],
        sessions: Optional[List[str]],
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        fraction = config.get("val_fraction")
        if fraction is None:
            return None, None

        canonical = self._canonical_type(config["type"])
        axis = str(config.get("split_axis", "subject")).lower()
        seed = int(config.get("split_seed", self.sampler_seed))
        data_root = str(config["data_root"])
        cache_key = (canonical, str(Path(data_root)), axis, float(fraction), seed)

        if cache_key not in self._split_cache:
            if axis == "subject":
                groups = subjects or self._dataset_subjects(canonical, data_root)
                prefix = "sub-"
            elif axis == "session":
                groups = sessions or self._dataset_sessions(
                    canonical,
                    data_root,
                    subjects=subjects,
                )
                prefix = "ses-"
            else:
                raise ValueError(
                    f"Unsupported split_axis={axis!r} for {config['type']}; "
                    "expected 'subject' or 'session'."
                )

            normalized = [_norm_id(group, prefix) for group in groups]
            self._split_cache[cache_key] = _fraction_holdout(
                normalized,
                fraction=float(fraction),
                seed=seed,
            )
            print(
                f"Deterministic validation split for {config['type']}: "
                f"axis={axis}, fraction={float(fraction):.3f}, seed={seed}, "
                f"groups={self._split_cache[cache_key]}"
            )

        selected = list(self._split_cache[cache_key])
        return (selected, None) if axis == "subject" else (None, selected)

    def _split_filters(
        self,
        config: Dict[str, Any],
        split: str,
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        canonical = self._canonical_type(config["type"])
        data_root = str(config["data_root"])
        subjects = _as_list(config.get("subjects"))
        sessions = _as_list(config.get("sessions"))
        val_subjects = _as_list(config.get("val_subjects"))
        val_sessions = _as_list(config.get("val_sessions"))

        if config.get("val_fraction") is not None:
            if val_subjects or val_sessions:
                raise ValueError(
                    f"Dataset {config['type']} defines val_fraction together with "
                    "an explicit validation split; choose one method."
                )
            val_subjects, val_sessions = self._fraction_split_groups(
                config,
                subjects=subjects,
                sessions=sessions,
            )

        if val_subjects and val_sessions:
            raise ValueError(
                f"Dataset {config['type']} defines both val_subjects and "
                "val_sessions; use only one split axis."
            )
        if not val_subjects and not val_sessions:
            raise ValueError(
                f"Missing validation split for {config['type']}. Add "
                "val_subjects=[...], val_sessions=[...], or val_fraction=... ."
            )

        if val_subjects:
            val_subjects = [_norm_id(item, "sub-") for item in val_subjects]
            if subjects is None:
                subjects = self._dataset_subjects(canonical, data_root)
            subjects = [_norm_id(item, "sub-") for item in subjects]
            if split == "val":
                return val_subjects, sessions
            return _exclude(subjects, val_subjects, "sub-"), sessions

        assert val_sessions is not None
        val_sessions = [_norm_id(item, "ses-") for item in val_sessions]
        if sessions is None:
            sessions = self._dataset_sessions(
                canonical,
                data_root,
                subjects=subjects,
            )
        sessions = [_norm_id(item, "ses-") for item in sessions]
        if split == "val":
            return subjects, val_sessions
        return subjects, _exclude(sessions, val_sessions, "ses-")

    def _create_dataset(
        self,
        config: Dict[str, Any],
        split: str,
        max_channel_dim: Optional[int],
    ):
        canonical = self._canonical_type(config["type"])
        subjects, sessions = self._split_filters(config, split=split)
        tasks = config.get("tasks", self._default_tasks(canonical))

        if self.debug_mode:
            if subjects:
                subjects = subjects[:1]
            if sessions:
                sessions = sessions[:1]

        dataset_classes = {
            "openneuro_ds004408": OpenNeuroEEGContinuousDataset,
            "openneuro_ds007808": OpenNeuroEEGContinuousDataset,
            "sparrkulee": SparrKULeeEEGContinuousDataset,
            "eegdash": EEGDashEEGContinuousDataset,
            "zuco": ZuCoEEGContinuousDataset,
        }
        dataset_class = dataset_classes[canonical]

        return dataset_class(
            data_root=config["data_root"],
            dataset_name=config.get("dataset_name", canonical),
            segment_length=config.get("segment_length", self.segment_length),
            subsegment_duration=config.get(
                "subsegment_duration", self.subsegment_duration
            ),
            cache_dir=config.get("cache_dir", self.cache_dir),
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            l_freq=config.get("l_freq", self.l_freq),
            h_freq=config.get("h_freq", self.h_freq),
            target_sfreq=config.get("target_sfreq", self.target_sfreq),
            channel_filter=config.get("channel_filter"),
            max_channel_dim=max_channel_dim,
            baseline_duration=config.get("baseline_duration", 0.5),
            clip_range=tuple(config.get("clip_range", (-5.0, 5.0))),
            listeningcovert_policy=config.get(
                "listeningcovert_policy", "listening_only"
            ),
            listening_trial_type=config.get("listening_trial_type", "listening"),
            listening_interval_start=config.get(
                "listening_interval_start", "onset"
            ),
            group_listeningcovert_by=config.get(
                "group_listeningcovert_by", "subject_session"
            ),
            cover_all_samples=config.get("cover_all_samples", True),
            short_stream_policy=config.get("short_stream_policy", "repeat"),
            merge_gap_seconds=config.get("merge_gap_seconds", 0.0),
        )

    def _subsample_by_recordings(self, dataset, proportion: float, seed: int):
        recording_ids = sorted(set(rec_idx for rec_idx, _ in dataset.segment_index))
        n_keep = max(1, int(round(len(recording_ids) * proportion)))
        rng = np.random.RandomState(seed)
        selected = set(
            int(value)
            for value in rng.choice(recording_ids, size=n_keep, replace=False).tolist()
        )
        print(
            f"Recording subsampling: {n_keep}/{len(recording_ids)} virtual streams "
            f"({proportion * 100:.1f}%)"
        )
        return SubsampledRecordingDataset(dataset, selected)

    @staticmethod
    def collate_fn(batch):
        eeg = torch.stack([item["meg"] for item in batch])
        sensor_xyzdir = torch.stack([item["sensor_xyzdir"] for item in batch])
        sensor_types = torch.stack([item["sensor_types"] for item in batch])
        sensor_mask = torch.stack([item["sensor_mask"] for item in batch])
        dataset_ids = torch.tensor(
            [item["dataset_idx"] for item in batch], dtype=torch.long
        )
        return eeg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test", "predict"):
            return

        max_channel_dim = self._resolve_max_channel_dim()

        if stage in (None, "fit"):
            train_datasets = []
            train_names = []
            for config in self.datasets_config:
                if config.get("validation_only", False):
                    continue
                dataset = self._create_dataset(
                    config, split="train", max_channel_dim=max_channel_dim
                )
                train_datasets.append(dataset)
                train_names.append(self._canonical_type(config["type"]))
                print(f"Training {config['type']}: {len(dataset)} continuous segments")

            if not train_datasets:
                raise ValueError(
                    "No training datasets configured. Remove validation_only=True "
                    "from at least one config."
                )

            self.train_dataset = MultiEEGDataset(train_datasets, train_names)
            if self.recording_subsample_prop is not None:
                self.train_dataset = self._subsample_by_recordings(
                    self.train_dataset,
                    proportion=self.recording_subsample_prop,
                    seed=self.sampler_seed,
                )
            print(f"Total continuous EEG training segments: {len(self.train_dataset)}")

        if stage in (None, "fit", "validate", "test"):
            val_datasets = []
            val_names = []
            for config in self.datasets_config:
                dataset = self._create_dataset(
                    config, split="val", max_channel_dim=max_channel_dim
                )
                val_datasets.append(dataset)
                val_names.append(self._canonical_type(config["type"]))
                print(f"Validation {config['type']}: {len(dataset)} continuous segments")

            self.val_dataset = MultiEEGDataset(val_datasets, val_names)
            if self.debug_mode:
                total = len(self.val_dataset)
                subset_size = max(1, int(round(0.1 * total)))
                rng = np.random.RandomState(self.sampler_seed)
                indices = sorted(
                    int(value)
                    for value in rng.choice(total, size=subset_size, replace=False)
                )
                self.val_dataset = Subset(self.val_dataset, indices)
                print(f"Debug mode: using {subset_size}/{total} validation segments")
            print(f"Total continuous EEG validation segments: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader().")

        sampler = None
        shuffle = True
        if self.use_recording_sampler:
            from .samplers import RecordingShuffleSampler

            sampler = RecordingShuffleSampler(
                self.train_dataset,
                seed=self.sampler_seed,
            )
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') or setup('validate') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        for dataset in (self.train_dataset, self.val_dataset):
            close = getattr(dataset, "close", None)
            if callable(close):
                close()

    def get_dataset_name_mapping(self) -> Dict[int, str]:
        dataset = (
            self.val_dataset.dataset
            if isinstance(self.val_dataset, Subset)
            else self.val_dataset
        )

        if dataset is None or not hasattr(dataset, "dataset_names"):
            return {}

        return {
            idx: name
            for idx, name in enumerate(dataset.dataset_names)
        }


__all__ = ["MultiEEGDataModule"]
