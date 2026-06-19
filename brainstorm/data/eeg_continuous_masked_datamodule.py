"""DataModule for the minimal continuous-EEG adaptation of MEG-XL."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .eeg_continuous_masked_dataset import (
    ContinuityAwareEEGDashDataset,
    ContinuityAwareOpenNeuroEEGDataset,
    ContinuityAwareSparrKULeeEEGDataset,
    ContinuityAwareZuCoEEGDataset,
    LISTENING_TARGET_POLICY,
)
from .eeg_continuous_multi_datamodule import (
    MultiEEGDataModule as LegacyContinuousEEGDataModule,
)


class MultiEEGDataModule(LegacyContinuousEEGDataModule):
    """Use complete physical runs and provide an optional target-time mask."""

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
            "openneuro_ds004408": ContinuityAwareOpenNeuroEEGDataset,
            "openneuro_ds007808": ContinuityAwareOpenNeuroEEGDataset,
            "sparrkulee": ContinuityAwareSparrKULeeEEGDataset,
            "eegdash": ContinuityAwareEEGDashDataset,
            "zuco": ContinuityAwareZuCoEEGDataset,
        }
        dataset_class = dataset_classes[canonical]

        return dataset_class(
            data_root=config["data_root"],
            dataset_name=config.get("dataset_name", canonical),
            segment_length=config.get("segment_length", self.segment_length),
            subsegment_duration=config.get(
                "subsegment_duration",
                self.subsegment_duration,
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
                "listeningcovert_policy",
                LISTENING_TARGET_POLICY,
            ),
            listening_trial_type=config.get(
                "listening_trial_type",
                "listening",
            ),
            listening_interval_start=config.get(
                "listening_interval_start",
                "onset",
            ),
            # These values are intentionally fixed to MEG-XL-style complete
            # non-overlapping windows. No run concatenation, overlap, repetition,
            # or temporal padding is permitted by the adapted dataset.
            group_listeningcovert_by="recording",
            cover_all_samples=False,
            short_stream_policy="error",
            merge_gap_seconds=config.get("merge_gap_seconds", 0.0),
        )

    @staticmethod
    def collate_fn(batch):
        eeg = torch.stack([item["meg"] for item in batch])
        sensor_xyzdir = torch.stack(
            [item["sensor_xyzdir"] for item in batch]
        )
        sensor_types = torch.stack(
            [item["sensor_types"] for item in batch]
        )
        sensor_mask = torch.stack(
            [item["sensor_mask"] for item in batch]
        )
        target_mask = torch.stack(
            [item["target_mask"] for item in batch]
        )
        dataset_ids = torch.tensor(
            [item["dataset_idx"] for item in batch],
            dtype=torch.long,
        )

        return (
            eeg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            target_mask,
            dataset_ids,
        )


__all__ = ["MultiEEGDataModule"]
