"""Minimal EEG data adaptation for MEG-XL continuous pre-training.

The model still receives fixed 150-second windows exactly as in MEG-XL. This
module only changes how EEG recordings are exposed to that model:

- every physical run remains an independent continuous stream;
- incomplete final windows are discarded rather than overlapped, repeated, or
  zero-padded;
- complete ``listeningcovert`` runs remain visible as context, while only
  intervals labelled ``listening`` are eligible reconstruction targets;
- EEG electrode positions are retained, but the MEG-like orientation vectors
  invented by legacy loaders are removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch

from .eegdash_eeg_continuous_dataset import EEGDashEEGContinuousDataset
from .openneuro_eeg_continuous_dataset import (
    EEG_SENSOR_TYPE_ID,
    IntervalRef,
    OpenNeuroEEGContinuousDataset,
)
from .sparrkulee_eeg_continuous_dataset import SparrKULeeEEGContinuousDataset
from .zuco_eeg_continuous_dataset import ZuCoEEGContinuousDataset


LISTENING_TARGET_POLICY = "full_recording_listening_targets"
_CACHE_SPATIAL_POLICY = "eeg_position_without_orientation_v1"


class ContinuityAwareEEGMixin:
    """Keep MEG-XL sampling semantics while adapting EEG run metadata."""

    def __init__(
        self,
        *args: Any,
        listeningcovert_policy: str = LISTENING_TARGET_POLICY,
        cover_all_samples: bool = False,
        short_stream_policy: str = "error",
        **kwargs: Any,
    ) -> None:
        if cover_all_samples:
            raise ValueError(
                "Continuity-aware EEG uses only complete non-overlapping windows; "
                "set cover_all_samples=false."
            )
        if str(short_stream_policy).lower() != "error":
            raise ValueError(
                "Continuity-aware EEG does not repeat or pad signal. Set "
                "short_stream_policy='error'; incomplete windows are discarded "
                "before reading."
            )

        super().__init__(
            *args,
            listeningcovert_policy=listeningcovert_policy,
            cover_all_samples=False,
            short_stream_policy="error",
            **kwargs,
        )

    def _validate_configuration(self) -> None:
        """Allow the target-only policy through the legacy configuration check."""
        requested_policy = self.listeningcovert_policy
        if requested_policy == LISTENING_TARGET_POLICY:
            self.listeningcovert_policy = "full_recording"
        try:
            super()._validate_configuration()
        finally:
            self.listeningcovert_policy = requested_policy

    def _cache_path(self, recording: Dict[str, Any]) -> Path:
        """Version caches because legacy EEG caches contain radial directions."""
        path = super()._cache_path(recording)
        return path.with_name(f"{path.stem}_{_CACHE_SPATIAL_POLICY}{path.suffix}")

    def _cache_is_readable(self, path: Path) -> bool:
        if not super()._cache_is_readable(path):
            return False
        try:
            with h5py.File(path, "r") as h5_file:
                return (
                    str(h5_file.attrs.get("eeg_spatial_policy", ""))
                    == _CACHE_SPATIAL_POLICY
                )
        except Exception:
            return False

    def _preprocess_recording(self, recording: Dict[str, Any]) -> None:
        super()._preprocess_recording(recording)
        with h5py.File(recording["cache_path"], "r+") as h5_file:
            sensor_xyzdir = h5_file["sensor_xyzdir"]
            if sensor_xyzdir.shape[1] >= 6:
                sensor_xyzdir[:, 3:6] = 0.0
            h5_file.attrs["eeg_spatial_policy"] = _CACHE_SPATIAL_POLICY

    @staticmethod
    def _sensor_xyzdir(raw) -> np.ndarray:
        """Keep normalized electrode XYZ and remove MEG coil orientation."""
        sensor_xyzdir = OpenNeuroEEGContinuousDataset._sensor_xyzdir(raw)
        sensor_xyzdir[:, 3:] = 0.0
        return sensor_xyzdir

    def _build_virtual_streams(self) -> List[Dict[str, Any]]:
        """Create exactly one virtual stream for each physical source run."""
        if self.listeningcovert_policy != LISTENING_TARGET_POLICY:
            return super()._build_virtual_streams()

        streams: List[Dict[str, Any]] = []
        for source_idx, recording in enumerate(self.source_recordings):
            metadata = self._source_metadata(source_idx)
            n_samples = int(metadata["n_samples"])
            sample_freq = float(metadata["sample_freq"])
            task = str(recording["task"]).lower()

            if task == "listeningcovert":
                target_ranges = self._listening_intervals(
                    recording,
                    n_samples,
                    sample_freq,
                )
                content_mode = LISTENING_TARGET_POLICY
            else:
                target_ranges = [(0, n_samples)]
                content_mode = "full_recording"

            streams.append(
                self._finalize_stream(
                    {
                        "subject": recording["subject"],
                        "sessions": {recording["session"]},
                        "task": recording["task"],
                        "runs": [recording["run"]],
                        "source_indices": [source_idx],
                        "intervals": [IntervalRef(source_idx, 0, n_samples)],
                        "sensor_signature": metadata["sensor_signature"],
                        "content_mode": content_mode,
                        "target_ranges": target_ranges,
                    }
                )
            )

        return streams

    @staticmethod
    def _has_contiguous_target(
        target_ranges: List[Tuple[int, int]],
        window_start: int,
        window_end: int,
        minimum_samples: int,
    ) -> bool:
        return any(
            min(end, window_end) - max(start, window_start) >= minimum_samples
            for start, end in target_ranges
        )

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        """Use only complete, non-overlapping windows inside a single run."""
        segment_samples = int(round(self.segment_length * self.target_sfreq))
        minimum_target_samples = int(
            round(self.subsegment_duration * self.target_sfreq)
        )
        segment_index: List[Tuple[int, int]] = []

        for stream_idx, stream in enumerate(self.recordings):
            total_samples = int(stream["total_samples"])
            target_ranges = list(
                stream.get("target_ranges", [(0, total_samples)])
            )

            starts = [
                start
                for start in range(
                    0,
                    max(0, total_samples - segment_samples + 1),
                    segment_samples,
                )
                if self._has_contiguous_target(
                    target_ranges,
                    start,
                    start + segment_samples,
                    minimum_target_samples,
                )
            ]

            self.segment_starts.append(starts)
            segment_index.extend(
                (stream_idx, segment_idx)
                for segment_idx in range(len(starts))
            )

        return segment_index

    def _print_summary(self) -> None:
        super()._print_summary()
        targeted = [
            stream
            for stream in self.recordings
            if stream["content_mode"] == LISTENING_TARGET_POLICY
        ]
        if targeted:
            target_samples = sum(
                end - start
                for stream in targeted
                for start, end in stream["target_ranges"]
            )
            print(
                f"{self.dataset_name}: complete listeningcovert runs retained; "
                f"reconstruction targets contain "
                f"{target_samples / self.target_sfreq / 3600.0:.3f} h of "
                f"trial_type={self.listening_trial_type!r}"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)

        stream_idx, segment_idx = self.segment_index[idx]
        stream = self.recordings[stream_idx]
        start = self.segment_starts[stream_idx][segment_idx]
        segment_samples = int(round(self.segment_length * self.target_sfreq))
        window_end = start + segment_samples

        target_mask = np.zeros(segment_samples, dtype=np.bool_)
        for target_start, target_end in stream.get(
            "target_ranges",
            [(0, int(stream["total_samples"]))],
        ):
            overlap_start = max(start, int(target_start))
            overlap_end = min(window_end, int(target_end))
            if overlap_end > overlap_start:
                target_mask[
                    overlap_start - start : overlap_end - start
                ] = True

        item["target_mask"] = torch.from_numpy(target_mask)

        valid_sensors = item["sensor_mask"].bool()
        if not torch.all(
            item["sensor_types"][valid_sensors] == EEG_SENSOR_TYPE_ID
        ):
            raise RuntimeError("Continuous EEG sample contains non-EEG sensors")

        return item


class ContinuityAwareOpenNeuroEEGDataset(
    ContinuityAwareEEGMixin,
    OpenNeuroEEGContinuousDataset,
):
    """OpenNeuro EEG with physical-run continuity and listening targets."""


class ContinuityAwareSparrKULeeEEGDataset(
    ContinuityAwareEEGMixin,
    SparrKULeeEEGContinuousDataset,
):
    """SparrKULee EEG with the same complete-run sampling semantics."""


class ContinuityAwareEEGDashDataset(
    ContinuityAwareEEGMixin,
    EEGDashEEGContinuousDataset,
):
    """EEGDash reading EEG with complete physical-run sampling semantics."""


class ContinuityAwareZuCoEEGDataset(
    ContinuityAwareEEGMixin,
    ZuCoEEGContinuousDataset,
):
    """ZuCo natural-reading EEG with complete physical-run sampling semantics."""


__all__ = [
    "LISTENING_TARGET_POLICY",
    "ContinuityAwareEEGMixin",
    "ContinuityAwareOpenNeuroEEGDataset",
    "ContinuityAwareSparrKULeeEEGDataset",
    "ContinuityAwareEEGDashDataset",
    "ContinuityAwareZuCoEEGDataset",
]
