"""Corrected Alice EEG loader used by the downstream word-decoding experiment.

This loader addresses two properties of the distributed Alice recordings:

* S26, S34, S35 and S36 have no event for audio segment 1; their eleven
  chronological markers correspond to segments 2--12.
* VEOG, Aux5 and AUD are auxiliary channels, not scalp EEG electrodes.  Aux5
  was previously retained, which left a channel without valid montage
  coordinates and propagated NaNs through sensor-position normalisation.

The cache namespace is deliberately versioned so caches created with the old
channel selection are never reused.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import mne
import numpy as np
import torch

from .alice_eeg_word_aligned_dataset import AliceEEGWordAlignedDataset
from .preprocessing import (
    cache_preprocessed,
    get_cache_path,
    is_hdf5_cache_readable,
)


class AliceEEGWordAlignedDatasetMissingFirstFix(AliceEEGWordAlignedDataset):
    """Alice loader with corrected markers, channel selection and validation."""

    MISSING_FIRST_EVENT_SUBJECTS = frozenset({"S26", "S34", "S35", "S36"})
    CACHE_CHANNEL_VERSION = "EEG_only_no_VEOG_AUX_AUD_finite_geometry_v3"

    @staticmethod
    def _is_auxiliary_channel(name: str) -> bool:
        key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
        return (
            key in {
                "VEOG",
                "HEOG",
                "EOG",
                "AUD",
                "AUDIO",
                "AUDIOIN",
                "AUX",
                "AUX5",
            }
            or key.startswith(("EOG", "AUD", "AUX"))
        )

    def _read_segment_onsets(self, raw_path: Path) -> Dict[int, float]:
        raw = mne.io.read_raw_brainvision(str(raw_path), preload=False, verbose=False)
        try:
            annotations = sorted(
                zip(raw.annotations.onset, raw.annotations.description),
                key=lambda item: float(item[0]),
            )
            subject_match = self._SUBJECT_RE.search(raw_path.stem)
            subject = (
                f"S{int(subject_match.group('id')):02d}"
                if subject_match is not None
                else raw_path.stem
            )

            stimulus = [
                (float(onset), str(description))
                for onset, description in annotations
                if "stimulus" in str(description).lower()
            ]
            coded = []
            for onset, description in annotations:
                match = self._TRAILING_TRIGGER_RE.search(str(description))
                if match and 1 <= int(match.group(1)) <= 12:
                    coded.append((float(onset), str(description)))

            def valid_count(markers: List[tuple[float, str]]) -> bool:
                return len(markers) >= 12 or (
                    len(markers) == 11
                    and subject in self.MISSING_FIRST_EVENT_SUBJECTS
                )

            markers = stimulus if valid_count(stimulus) else coded
            if not valid_count(markers):
                expected = (
                    "11 markers for documented segments 2--12"
                    if subject in self.MISSING_FIRST_EVENT_SUBJECTS
                    else "12 markers for segments 1--12"
                )
                raise ValueError(
                    f"Expected {expected} in {raw_path}; found "
                    f"{len(markers)} usable markers."
                )

            if subject in self.MISSING_FIRST_EVENT_SUBJECTS and len(markers) == 11:
                segment_ids = list(range(2, 13))
                selected = markers
                warnings.warn(
                    f"{subject} has the documented missing first Alice event; "
                    "mapping its 11 chronological markers to segments 2--12 and "
                    "omitting segment 1."
                )
            else:
                segment_ids = list(range(1, 13))
                selected = markers[:12]
                if len(markers) > 12:
                    warnings.warn(
                        f"Found {len(markers)} candidate Alice markers in {raw_path}; "
                        "using the first 12 chronological markers."
                    )

            onsets = np.asarray([item[0] for item in selected], dtype=np.float64)
            if not np.all(np.diff(onsets) > 0):
                raise ValueError(
                    f"Alice segment markers are not strictly increasing in {raw_path}."
                )

            result: Dict[int, float] = {}
            for segment_id, onset in zip(segment_ids, onsets):
                lag = self.marker_lag_first if segment_id == 1 else self.marker_lag_other
                result[int(segment_id)] = float(onset + lag)
            return result
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()

    def _find_subject_recordings(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for raw_path in sorted(self.data_root.glob("S*.vhdr")):
            match = self._SUBJECT_RE.search(raw_path.stem)
            if not match:
                continue
            subject = f"S{int(match.group('id')):02d}"
            if self.subjects is not None and subject not in self.subjects:
                continue
            if subject in result:
                warnings.warn(f"Ignoring duplicate VHDR for {subject}: {raw_path}")
                continue

            filter_name = self.CACHE_CHANNEL_VERSION
            if self.channel_filter is not None:
                filter_name += f"_{getattr(self.channel_filter, '__name__', 'custom')}"
            filter_name += f"_type{self.eeg_sensor_type_id}"

            result[subject] = {
                "subject": subject,
                "session": "ses-001",
                "raw_path": raw_path,
                "segment_onsets": self._read_segment_onsets(raw_path),
                "filter_name": filter_name,
                "cache_path": get_cache_path(
                    self.cache_dir,
                    subject,
                    "ses-001",
                    "continuous",
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter_name=filter_name,
                    dataset_name=self.dataset_name,
                    task_mode=self.task_mode,
                    segment_length=self.segment_length,
                    subsegment_duration=self.subsegment_duration,
                    window_onset_offset=self.window_onset_offset,
                    tokenizer_name=self.tokenizer_name,
                ),
            }
        return result

    @staticmethod
    def _channel_has_finite_position(raw: mne.io.BaseRaw, name: str) -> bool:
        index = raw.ch_names.index(name)
        position = np.asarray(raw.info["chs"][index]["loc"][:3], dtype=np.float64)
        return bool(np.all(np.isfinite(position)) and np.linalg.norm(position) > 0)

    def _preprocess_subjects(self) -> None:
        total = len(self.subject_recordings)
        for index, record in enumerate(self.subject_recordings.values(), 1):
            if is_hdf5_cache_readable(record["cache_path"]):
                print(f"Using cached Alice recording {index}/{total}: {record['subject']}")
                continue

            print(f"Preprocessing Alice recording {index}/{total}: {record['subject']}")
            raw = mne.io.read_raw_brainvision(
                str(record["raw_path"]), preload=True, verbose=False
            )
            try:
                picks = mne.pick_types(
                    raw.info,
                    meg=False,
                    eeg=True,
                    eog=False,
                    ecg=False,
                    emg=False,
                    stim=False,
                    misc=False,
                    exclude=[],
                )
                keep = [
                    raw.ch_names[pick]
                    for pick in picks
                    if not self._is_auxiliary_channel(raw.ch_names[pick])
                ]
                if self.channel_filter is not None:
                    keep = [name for name in keep if self.channel_filter(name)]
                if not keep:
                    raise ValueError(f"No EEG channels remain in {record['raw_path']}.")

                raw.pick(keep)
                self._apply_montage(raw)

                invalid_geometry = [
                    name
                    for name in raw.ch_names
                    if not self._channel_has_finite_position(raw, name)
                ]
                if invalid_geometry:
                    warnings.warn(
                        f"Dropping Alice channels without finite electrode positions for "
                        f"{record['subject']}: {invalid_geometry}"
                    )
                    raw.drop_channels(invalid_geometry)

                if not raw.ch_names:
                    raise ValueError(
                        f"No Alice EEG channels with valid geometry remain for "
                        f"{record['subject']}."
                    )
                if len(raw.ch_names) > int(self.max_channel_dim or len(raw.ch_names)):
                    raise ValueError(
                        f"{record['subject']} retains {len(raw.ch_names)} EEG channels, "
                        f"exceeding max_channel_dim={self.max_channel_dim}."
                    )
                if not np.all(np.isfinite(raw.get_data())):
                    raise ValueError(
                        f"Non-finite raw EEG values found before filtering for "
                        f"{record['subject']}."
                    )

                if self.h_freq >= raw.info["sfreq"] / 2:
                    raise ValueError("h_freq must be below the original Nyquist frequency.")
                raw.filter(self.l_freq, self.h_freq, n_jobs=1, verbose=False)
                if not np.isclose(raw.info["sfreq"], self.target_sfreq):
                    raw.resample(self.target_sfreq, n_jobs=1, verbose=False)
                if not np.all(np.isfinite(raw.get_data())):
                    raise ValueError(
                        f"Non-finite EEG values found after preprocessing for "
                        f"{record['subject']}."
                    )

                cache_preprocessed(
                    raw,
                    record["cache_path"],
                    {
                        "subject": record["subject"],
                        "session": "ses-001",
                        "task": "alice_chapter_one",
                        "dataset": self.dataset_name,
                        "eeg_sensor_type_id": self.eeg_sensor_type_id,
                        "alice_channel_filter_version": self.CACHE_CHANNEL_VERSION,
                    },
                    l_freq=self.l_freq,
                    h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter_name=record["filter_name"],
                )
            finally:
                close = getattr(raw, "close", None)
                if callable(close):
                    close()
            print(
                f"  Cached {len(raw.ch_names)} finite-position EEG channels to "
                f"{record['cache_path']}"
            )

    def _open_subject_caches(self) -> None:
        super()._open_subject_caches()
        for subject, record in self.subject_recordings.items():
            xyzdir = np.asarray(record["sensor_xyzdir"], dtype=np.float32)
            if not np.all(np.isfinite(xyzdir)):
                self.close()
                raise ValueError(
                    f"Non-finite sensor geometry remains in the corrected Alice cache "
                    f"for {subject}: {record['cache_path']}"
                )
            positions = xyzdir[:, :3]
            scale = float(np.sqrt(3 * np.mean(np.sum((positions - positions.mean(0)) ** 2, axis=1))))
            if not np.isfinite(scale) or scale <= 0:
                self.close()
                raise ValueError(
                    f"Degenerate sensor geometry in the corrected Alice cache for "
                    f"{subject}: {record['cache_path']}"
                )

    def _make_word_groups(
        self, recording: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        segment_onsets = recording["segment_onsets"]

        for event in self.word_events.itertuples(index=False):
            segment_id = int(event.segment)
            if segment_id not in segment_onsets:
                current = []
                continue

            absolute_onset = float(segment_onsets[segment_id]) + float(event.onset)
            start = absolute_onset + self.window_onset_offset
            end = start + self.subsegment_duration
            if start < 0 or end > recording["duration"]:
                current = []
                continue

            current.append(
                {
                    "word": str(event.word),
                    "onset": absolute_onset,
                    "window_start": start,
                    "window_end": end,
                    "segment": segment_id,
                    "order": int(event.order),
                    "subsegment_idx": len(current),
                }
            )
            if len(current) == self.words_per_segment:
                groups.append(current.copy())
                current = []
        return groups

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        for key in ("meg", "sensor_xyzdir"):
            value = sample[key]
            if not torch.isfinite(value).all():
                n_bad = int((~torch.isfinite(value)).sum().item())
                raise FloatingPointError(
                    f"Alice sample {idx} ({sample['subject']}) contains {n_bad} "
                    f"non-finite values in {key}. Delete stale caches if this path "
                    "was created by an older loader."
                )
        return sample
