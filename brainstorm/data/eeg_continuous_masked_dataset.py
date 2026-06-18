"""Continuity-aware continuous EEG datasets.

The classes in this module keep each source run intact. For ds007808
``task-listeningcovert`` recordings, the complete run is visible to the model,
while only samples labelled ``listening`` are valid reconstruction targets.

They also provide:
- average EEG reference in the cached signal;
- zero orientation vectors for EEG electrodes;
- explicit time-valid and target-valid masks;
- zero padding for short recordings instead of cyclic repetition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch

from .openneuro_eeg_continuous_dataset import (
    EEG_SENSOR_TYPE_ID,
    IntervalRef,
    OpenNeuroEEGContinuousDataset,
    segment_starts_cover_all,
)
from .sparrkulee_eeg_continuous_dataset import SparrKULeeEEGContinuousDataset


LISTENING_TARGET_POLICY = "full_recording_listening_targets"


class ContinuityAwareEEGMixin:
    """Shared behavior for OpenNeuro and SparrKULee continuous EEG loaders."""

    eeg_reference: str

    def __init__(
        self,
        *args: Any,
        eeg_reference: str = "average",
        listeningcovert_policy: str = LISTENING_TARGET_POLICY,
        short_stream_policy: str = "zero_pad",
        **kwargs: Any,
    ) -> None:
        self.eeg_reference = str(eeg_reference).strip().lower()
        if self.eeg_reference not in {"average", "none"}:
            raise ValueError(
                "eeg_reference must be 'average' or 'none', "
                f"got {eeg_reference!r}"
            )
        super().__init__(
            *args,
            listeningcovert_policy=listeningcovert_policy,
            short_stream_policy=short_stream_policy,
            **kwargs,
        )

    def _validate_configuration(self) -> None:
        """Extend the legacy policy validation without changing the base loader."""
        requested_policy = self.listeningcovert_policy
        if requested_policy == LISTENING_TARGET_POLICY:
            self.listeningcovert_policy = "full_recording"
        try:
            super()._validate_configuration()
        finally:
            self.listeningcovert_policy = requested_policy

        if self.short_stream_policy == "repeat":
            raise ValueError(
                "Cyclic repetition is disabled for continuity-aware EEG. "
                "Use short_stream_policy='zero_pad' or 'error'."
            )

    def _cache_path(self, recording: Dict[str, Any]) -> Path:
        path = super()._cache_path(recording)
        suffix = "avgref_v1" if self.eeg_reference == "average" else "native_ref_v1"
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")

    def _cache_is_readable(self, path: Path) -> bool:
        if not super()._cache_is_readable(path):
            return False
        try:
            with h5py.File(path, "r") as h5_file:
                return str(h5_file.attrs.get("eeg_reference", "")) == self.eeg_reference
        except Exception:
            return False

    def _preprocess_recording(self, recording: Dict[str, Any]) -> None:
        """Run shared preprocessing and apply a common average reference.

        Average referencing is linear and therefore commutes with the filtering
        and resampling performed by the base loader. Applying it to the cached
        data avoids duplicating the complete format-specific preprocessing path.
        """
        super()._preprocess_recording(recording)
        cache_path = Path(recording["cache_path"])

        with h5py.File(cache_path, "r+") as h5_file:
            data = h5_file["data"]
            if self.eeg_reference == "average":
                chunk_samples = max(1, int(round(self.target_sfreq * 60.0)))
                for start in range(0, data.shape[1], chunk_samples):
                    end = min(data.shape[1], start + chunk_samples)
                    block = np.asarray(data[:, start:end], dtype=np.float32)
                    block -= block.mean(axis=0, keepdims=True)
                    data[:, start:end] = block
            h5_file.attrs["eeg_reference"] = self.eeg_reference

    @staticmethod
    def _sensor_xyzdir(raw) -> np.ndarray:
        """Keep EEG positions but do not invent a MEG-like coil orientation."""
        sensor_xyzdir = OpenNeuroEEGContinuousDataset._sensor_xyzdir(raw)
        sensor_xyzdir[:, 3:] = 0.0
        return sensor_xyzdir

    def _build_virtual_streams(self) -> List[Dict[str, Any]]:
        """Create one stream per physical run.

        For listeningcovert, target_ranges marks listening samples while the
        stream itself remains the complete, temporally continuous recording.
        """
        if self.listeningcovert_policy != LISTENING_TARGET_POLICY:
            return super()._build_virtual_streams()

        streams: List[Dict[str, Any]] = []
        for source_idx, recording in enumerate(self.source_recordings):
            metadata = self._source_metadata(source_idx)
            n_samples = int(metadata["n_samples"])
            sample_freq = float(metadata["sample_freq"])
            task = recording["task"].lower()

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

            stream = self._finalize_stream(
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
            streams.append(stream)
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
        """Build windows but discard those without one complete target block."""
        segment_samples = int(round(self.segment_length * self.target_sfreq))
        minimum_target_samples = int(
            round(self.subsegment_duration * self.target_sfreq)
        )
        segment_index: List[Tuple[int, int]] = []

        for stream_idx, stream in enumerate(self.recordings):
            total_samples = int(stream["total_samples"])
            if total_samples <= 0:
                self.segment_starts.append([])
                continue

            if total_samples < segment_samples and self.short_stream_policy == "error":
                raise ValueError(
                    f"Recording {stream['recording_name']} has only "
                    f"{total_samples / self.target_sfreq:.2f}s, shorter than "
                    f"segment_length={self.segment_length}s"
                )

            if self.cover_all_samples:
                candidates = segment_starts_cover_all(
                    total_samples,
                    segment_samples,
                )
            else:
                candidates = list(
                    range(
                        0,
                        max(0, total_samples - segment_samples + 1),
                        segment_samples,
                    )
                )
                if total_samples < segment_samples:
                    candidates = [0]

            target_ranges = list(
                stream.get("target_ranges", [(0, total_samples)])
            )
            starts = [
                start
                for start in candidates
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
                f"loss targets {target_samples / self.target_sfreq / 3600.0:.3f} h "
                f"of trial_type={self.listening_trial_type!r}"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)

        stream_idx, segment_idx = self.segment_index[idx]
        stream = self.recordings[stream_idx]
        start = self.segment_starts[stream_idx][segment_idx]
        segment_samples = int(round(self.segment_length * self.target_sfreq))
        total_samples = int(stream["total_samples"])

        valid_samples = max(0, min(segment_samples, total_samples - start))
        time_mask = np.zeros(segment_samples, dtype=np.bool_)
        time_mask[:valid_samples] = True

        target_mask = np.zeros(segment_samples, dtype=np.bool_)
        window_end = start + segment_samples
        for target_start, target_end in stream.get(
            "target_ranges",
            [(0, total_samples)],
        ):
            overlap_start = max(start, int(target_start))
            overlap_end = min(window_end, int(target_end))
            if overlap_end > overlap_start:
                target_mask[
                    overlap_start - start : overlap_end - start
                ] = True
        target_mask &= time_mask

        item["time_mask"] = torch.from_numpy(time_mask)
        item["target_mask"] = torch.from_numpy(target_mask)
        item["eeg_reference"] = self.eeg_reference

        # EEG uses one modality ID and no MEG coil orientation. Keep the legacy
        # sensor_type value 2 so existing metadata and checkpoints remain legible.
        if not torch.all(
            item["sensor_types"][item["sensor_mask"].bool()]
            == EEG_SENSOR_TYPE_ID
        ):
            raise RuntimeError("Continuity-aware EEG sample contains non-EEG sensors")
        return item


class ContinuityAwareOpenNeuroEEGDataset(
    ContinuityAwareEEGMixin,
    OpenNeuroEEGContinuousDataset,
):
    """OpenNeuro continuous EEG with true run continuity and target masks."""


class ContinuityAwareSparrKULeeEEGDataset(
    ContinuityAwareEEGMixin,
    SparrKULeeEEGContinuousDataset,
):
    """SparrKULee continuous EEG with shared reference and mask semantics."""


__all__ = [
    "LISTENING_TARGET_POLICY",
    "ContinuityAwareOpenNeuroEEGDataset",
    "ContinuityAwareSparrKULeeEEGDataset",
]
