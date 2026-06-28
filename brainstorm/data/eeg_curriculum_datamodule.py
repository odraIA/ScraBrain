"""DataModule for the two-stage EEG language curriculum.

It extends the continuity-aware EEG pre-training loader with the word-aligned
Alice and Weissbart datasets. Their complete 150-second examples are valid
masked-token targets, while ds007808 keeps its listening-only target mask inside
listeningcovert recordings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from brainstorm.data.alice_eeg_word_aligned_dataset import (
    AliceEEGWordAlignedDataset,
)
from brainstorm.data.eeg_continuous_masked_datamodule import (
    MultiEEGDataModule as ContinuityAwareEEGDataModule,
)
from brainstorm.data.weissbart_eeg_word_aligned_dataset import (
    WeissbartEEGWordAlignedDataset,
)
from brainstorm.models.criss_cross_transformer import EEG_SENSOR_TYPE_ID


_SPECIAL_ALIASES = {
    "alice": "alice_eeg",
    "alice_eeg": "alice_eeg",
    "weissbart": "weissbart_eeg",
    "weissbart_eeg": "weissbart_eeg",
}


class CurriculumEEGDataModule(ContinuityAwareEEGDataModule):
    """Combine reading/listening EEG datasets without including ds004408."""

    @staticmethod
    def _canonical_type(dataset_type: str) -> str:
        key = str(dataset_type).strip().lower()
        if key in _SPECIAL_ALIASES:
            return _SPECIAL_ALIASES[key]
        return ContinuityAwareEEGDataModule._canonical_type(dataset_type)

    def _default_tasks(self, dataset_type: str):
        canonical = self._canonical_type(dataset_type)
        if canonical == "alice_eeg":
            return ["listening"]
        if canonical == "weissbart_eeg":
            return list(WeissbartEEGWordAlignedDataset.STORY_NAMES)
        return super()._default_tasks(canonical)

    def _dataset_subjects(self, canonical: str, data_root: str):
        if canonical == "alice_eeg":
            root = AliceEEGWordAlignedDataset._resolve_root(Path(data_root))
            subjects = []
            for path in sorted(root.glob("S*.vhdr")):
                match = AliceEEGWordAlignedDataset._SUBJECT_RE.search(path.stem)
                if match:
                    subjects.append(f"S{int(match.group('id')):02d}")
            return subjects

        if canonical == "weissbart_eeg":
            root = WeissbartEEGWordAlignedDataset._resolve_root(Path(data_root))
            subjects = set()
            for path in sorted((root / "eeg").rglob("*.vhdr")):
                match = re.search(r"P(?P<id>\d{1,2})", str(path), re.IGNORECASE)
                if match:
                    subjects.add(f"P{int(match.group('id')):02d}")
            return sorted(subjects)

        return super()._dataset_subjects(canonical, data_root)

    def _dataset_sessions(
        self,
        canonical: str,
        data_root: str,
        subjects,
    ):
        if canonical in {"alice_eeg", "weissbart_eeg"}:
            return ["ses-001"]
        return super()._dataset_sessions(canonical, data_root, subjects)

    def _create_dataset(
        self,
        config: Dict[str, Any],
        split: str,
        max_channel_dim: Optional[int],
    ):
        canonical = self._canonical_type(config["type"])
        if canonical not in {"alice_eeg", "weissbart_eeg"}:
            return super()._create_dataset(config, split, max_channel_dim)

        subjects, sessions = self._split_filters(config, split=split)
        tasks = config.get("tasks", self._default_tasks(canonical))
        if self.debug_mode:
            if subjects:
                subjects = subjects[:1]
            if sessions:
                sessions = sessions[:1]

        common = {
            "data_root": config["data_root"],
            "segment_length": config.get("segment_length", self.segment_length),
            "subsegment_duration": config.get(
                "subsegment_duration", self.subsegment_duration
            ),
            "words_per_segment": config.get(
                "words_per_segment", self.words_per_segment
            ),
            "window_onset_offset": config.get(
                "window_onset_offset", self.window_onset_offset
            ),
            "cache_dir": config.get(
                "cache_dir", str(Path(self.cache_dir) / canonical)
            ),
            "subjects": subjects,
            "sessions": sessions,
            "tasks": tasks,
            "l_freq": config.get("l_freq", self.l_freq),
            "h_freq": config.get("h_freq", self.h_freq),
            "target_sfreq": config.get("target_sfreq", self.target_sfreq),
            "channel_filter": config.get("channel_filter"),
            "max_channel_dim": max_channel_dim,
            "baseline_duration": config.get("baseline_duration", 0.5),
            "clip_range": tuple(config.get("clip_range", (-5.0, 5.0))),
            "eeg_sensor_type": config.get("eeg_sensor_type", "eeg"),
            "dataset_name": config.get("dataset_name", canonical),
            "task_mode": config.get("task_mode", "listening"),
            "tokenizer_name": config.get("tokenizer_name", self.tokenizer_name),
        }

        if canonical == "alice_eeg":
            return AliceEEGWordAlignedDataset(
                **common,
                subject_selection=config.get("subject_selection", "main"),
                marker_lag_first=config.get("marker_lag_first", 0.060),
                marker_lag_other=config.get("marker_lag_other", 0.050),
            )
        return WeissbartEEGWordAlignedDataset(**common)

    @staticmethod
    def collate_fn(batch):
        eeg = torch.stack([item["meg"] for item in batch])
        sensor_xyzdir = torch.stack([item["sensor_xyzdir"] for item in batch])
        sensor_types = torch.stack([item["sensor_types"] for item in batch])
        sensor_mask = torch.stack([item["sensor_mask"] for item in batch])

        target_masks = []
        for item in batch:
            valid_sensors = item["sensor_mask"].bool()
            if not torch.all(
                item["sensor_types"][valid_sensors].long() == EEG_SENSOR_TYPE_ID
            ):
                raise RuntimeError(
                    "Curriculum sample contains a valid sensor not marked as EEG type 2"
                )

            target_mask = item.get("target_mask")
            if target_mask is None:
                target_mask = torch.ones(item["meg"].shape[-1], dtype=torch.bool)
            target_masks.append(target_mask.bool())

        dataset_ids = torch.tensor(
            [item["dataset_idx"] for item in batch],
            dtype=torch.long,
        )
        return (
            eeg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            torch.stack(target_masks),
            dataset_ids,
        )


__all__ = ["CurriculumEEGDataModule"]
