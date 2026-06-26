"""MEG-XL-compatible word-aligned loader for the Alice listening EEG dataset."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    _process_single_chunk,
    cache_preprocessed,
    get_cache_path,
    is_hdf5_cache_readable,
    load_cached,
)
from .utils import norm_sensor_positions


class AliceEEGWordAlignedDataset(Dataset):
    """Expose Brennan's Alice EEG data using the MEG-XL word-decoding contract.

    The same audiobook chapter is repeated across participants. To avoid textual
    leakage, callers should split with ``split_group=sentence`` so every repeated
    50-word sequence is assigned to exactly one of train, validation, or test.

    Word times in ``AliceChapterOne-EEG.csv`` are relative to one of twelve audio
    segments. Their absolute EEG times are reconstructed from the BrainVision
    stimulus markers, including the 60 ms delay correction for segment 1 and the
    50 ms correction for segments 2--12 used by the original analysis code.
    """

    MAIN_SUBJECTS = (
        "S01", "S03", "S04", "S05", "S06", "S08", "S10", "S11", "S12",
        "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20", "S21",
        "S22", "S25", "S26", "S34", "S35", "S36", "S37", "S38", "S39",
        "S40", "S41", "S42", "S44", "S45", "S48",
    )
    LOW_PERFORMANCE_SUBJECTS = (
        "S07", "S09", "S23", "S24", "S27", "S30", "S32", "S43",
    )
    HIGH_NOISE_SUBJECTS = (
        "S02", "S28", "S29", "S31", "S33", "S46", "S47", "S49",
    )
    _SUBJECT_RE = re.compile(r"S(?P<id>\d{2})", re.IGNORECASE)
    _TRAILING_TRIGGER_RE = re.compile(r"(?:^|[/\s])S?\s*(\d+)\s*$", re.IGNORECASE)

    def __init__(
        self,
        data_root: str,
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Optional[Callable[[str], bool]] = None,
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5),
        eeg_sensor_type: str = "grad",
        dataset_name: str = "alice_eeg",
        task_mode: str = "listening",
        tokenizer_name: str = "biocodec",
        subject_selection: str = "main",
        marker_lag_first: float = 0.060,
        marker_lag_other: float = 0.050,
    ):
        self.data_root = self._resolve_root(Path(data_root))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.segment_length = float(segment_length)
        self.subsegment_duration = float(subsegment_duration)
        self.words_per_segment = int(words_per_segment)
        self.window_onset_offset = float(window_onset_offset)
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.target_sfreq = float(target_sfreq)
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.baseline_duration = float(baseline_duration)
        self.clip_range = tuple(clip_range)
        self.eeg_sensor_type = eeg_sensor_type
        self.eeg_sensor_type_id = self._sensor_type_id(eeg_sensor_type)
        self.dataset_name = dataset_name
        self.task_mode = task_mode
        self.tokenizer_name = tokenizer_name
        self.subject_selection = str(subject_selection).strip().lower()
        self.marker_lag_first = float(marker_lag_first)
        self.marker_lag_other = float(marker_lag_other)

        expected = self.words_per_segment * self.subsegment_duration
        if not np.isclose(self.segment_length, expected):
            raise ValueError(
                "segment_length must equal words_per_segment * subsegment_duration "
                f"({self.segment_length} != {expected})."
            )
        if self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive.")

        self.subjects = self._resolve_subject_selection(subjects)
        self.sessions = (
            {self._normalise_session(value) for value in sessions}
            if sessions is not None else None
        )
        if self.sessions is not None and "ses-001" not in self.sessions:
            raise ValueError("Alice EEG only exposes the logical session 'ses-001'.")
        self.tasks = {str(value).strip().lower() for value in tasks} if tasks else None
        if self.tasks and not self.tasks.intersection(
            {"alice", "alice_chapter_one", "listening", "continuous"}
        ):
            raise ValueError(
                "Alice EEG exposes one listening task; use alice, alice_chapter_one, "
                "listening, or leave tasks unset."
            )

        self.word_events = self._read_word_events(
            self.data_root / "AliceChapterOne-EEG.csv"
        )
        unique_words = int(self.word_events["word"].nunique())
        if unique_words != 601:
            warnings.warn(
                "The cleaned Alice transcript contains "
                f"{unique_words} unique words rather than the 601 reported by the "
                "comparison paper. The evaluator will still use the configured "
                "601-word retrieval set, capped by the available vocabulary."
            )

        self.subject_recordings = self._find_subject_recordings()
        if not self.subject_recordings:
            raise ValueError(f"No Alice BrainVision files found under {self.data_root}.")

        self._preprocess_subjects()
        self.file_handles: Dict[str, h5py.File] = {}
        self._open_subject_caches()
        self.recordings = self._make_recordings()
        self.word_groups = [self._make_word_groups(recording) for recording in self.recordings]
        self.segment_index = [
            (recording_idx, group_idx)
            for recording_idx, groups in enumerate(self.word_groups)
            for group_idx in range(len(groups))
        ]

        for index, (recording, groups) in enumerate(
            zip(self.recordings, self.word_groups)
        ):
            print(
                f"Recording {index} ({recording['subject']} Alice chapter 1): "
                f"Found {len(groups)} word-aligned segments"
            )
        if not self.segment_index:
            self.close()
            raise ValueError(
                "No complete Alice word-aligned segments were created. Check the "
                "BrainVision markers/CSV alignment or reduce words_per_segment."
            )

    @staticmethod
    def _resolve_root(path: Path) -> Path:
        for candidate in (path, path / "alice_eeg"):
            if (
                (candidate / "AliceChapterOne-EEG.csv").exists()
                and any(candidate.glob("S*.vhdr"))
            ):
                return candidate
        raise FileNotFoundError(
            f"Expected AliceChapterOne-EEG.csv and S*.vhdr files under {path}."
        )

    @staticmethod
    def _sensor_type_id(value: str) -> int:
        aliases = {
            "grad": 0,
            "gradiometer": 0,
            "mag": 1,
            "meg": 1,
            "magnetometer": 1,
            "eeg": 2,
        }
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown eeg_sensor_type {value!r}; choose {sorted(aliases)}."
            )
        return aliases[key]

    @classmethod
    def _normalise_subject(cls, value: str) -> str:
        text = re.sub(r"^sub-", "", str(value).strip(), flags=re.IGNORECASE)
        match = cls._SUBJECT_RE.search(text)
        if match:
            return f"S{int(match.group('id')):02d}"
        return f"S{int(text):02d}" if text.isdigit() else text.upper()

    @staticmethod
    def _normalise_session(value: str) -> str:
        text = str(value).strip().lower()
        if text in {"1", "001", "ses-1", "ses-001", "session-1", "session-001"}:
            return "ses-001"
        return text

    def _resolve_subject_selection(
        self, subjects: Optional[List[str]]
    ) -> Optional[set[str]]:
        if subjects is not None:
            return {self._normalise_subject(value) for value in subjects}

        selections = {
            "main": set(self.MAIN_SUBJECTS),
            "paper": set(self.MAIN_SUBJECTS),
            "all": None,
            "low_performance": set(self.LOW_PERFORMANCE_SUBJECTS),
            "low_perf": set(self.LOW_PERFORMANCE_SUBJECTS),
            "high_noise": set(self.HIGH_NOISE_SUBJECTS),
        }
        if self.subject_selection not in selections:
            raise ValueError(
                f"Unknown subject_selection={self.subject_selection!r}; choose "
                f"{sorted(selections)} or pass data.subjects explicitly."
            )
        return selections[self.subject_selection]

    @staticmethod
    def _clean_word(value: Any) -> str:
        word = str(value).strip().strip('"').strip("'").lower()
        return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)

    def _read_word_events(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required = {"word", "segment", "onset"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"{path} needs columns {sorted(required)}; found {list(frame.columns)}."
            )

        events = pd.DataFrame(
            {
                "word": [self._clean_word(value) for value in frame["word"]],
                "segment": pd.to_numeric(frame["segment"], errors="coerce"),
                "onset": pd.to_numeric(frame["onset"], errors="coerce"),
                "order": pd.to_numeric(
                    frame["order"] if "order" in frame.columns else np.arange(len(frame)),
                    errors="coerce",
                ),
            }
        ).dropna(subset=["segment", "onset", "order"])
        events["segment"] = events["segment"].astype(int)
        events = events[
            events["segment"].between(1, 12)
            & ~events["word"].isin({"", "sp", "silence", "s"})
        ]
        return events.sort_values(["order", "segment", "onset"]).reset_index(drop=True)

    @staticmethod
    def _is_auxiliary_channel(name: str) -> bool:
        key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
        return key in {"VEOG", "HEOG", "EOG", "AUD", "AUDIO", "AUDIOIN"} or key.startswith(
            ("EOG", "AUD")
        )

    def _read_segment_onsets(self, raw_path: Path) -> np.ndarray:
        raw = mne.io.read_raw_brainvision(str(raw_path), preload=False, verbose=False)
        try:
            annotations = sorted(
                zip(raw.annotations.onset, raw.annotations.description),
                key=lambda item: float(item[0]),
            )
            stimulus = [
                (float(onset), str(description))
                for onset, description in annotations
                if "stimulus" in str(description).lower()
            ]
            if len(stimulus) < 12:
                coded = []
                for onset, description in annotations:
                    match = self._TRAILING_TRIGGER_RE.search(str(description))
                    if match and 1 <= int(match.group(1)) <= 12:
                        coded.append((float(onset), str(description)))
                stimulus = coded
            if len(stimulus) < 12:
                raise ValueError(
                    f"Expected 12 Alice audio-segment markers in {raw_path}; "
                    f"found {len(stimulus)}."
                )

            onsets = np.asarray([item[0] for item in stimulus[:12]], dtype=np.float64)
            if not np.all(np.diff(onsets) > 0):
                raise ValueError(f"Alice segment markers are not strictly increasing in {raw_path}.")
            onsets[0] += self.marker_lag_first
            onsets[1:] += self.marker_lag_other
            return onsets
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

            filter_name = "EEG_only_no_VEOG_AUD"
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

    def _apply_montage(self, raw: mne.io.BaseRaw) -> None:
        montage_path = self.data_root / "easycapM10-acti61_elec.sfp"
        try:
            if montage_path.exists():
                montage = mne.channels.read_custom_montage(str(montage_path))
                raw.set_montage(montage, on_missing="ignore", verbose=False)
            else:
                raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
        except (TypeError, ValueError, RuntimeError) as exc:
            warnings.warn(f"Could not apply Alice electrode montage: {exc}")

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

            if self.h_freq >= raw.info["sfreq"] / 2:
                raise ValueError("h_freq must be below the original Nyquist frequency.")
            raw.filter(self.l_freq, self.h_freq, n_jobs=1, verbose=False)
            if not np.isclose(raw.info["sfreq"], self.target_sfreq):
                raw.resample(self.target_sfreq, n_jobs=1, verbose=False)

            cache_preprocessed(
                raw,
                record["cache_path"],
                {
                    "subject": record["subject"],
                    "session": "ses-001",
                    "task": "alice_chapter_one",
                    "dataset": self.dataset_name,
                    "eeg_sensor_type_id": self.eeg_sensor_type_id,
                },
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                target_sfreq=self.target_sfreq,
                channel_filter_name=record["filter_name"],
            )
            close = getattr(raw, "close", None)
            if callable(close):
                close()
            print(f"  Cached to {record['cache_path']}")

    def _open_subject_caches(self) -> None:
        for subject, record in self.subject_recordings.items():
            handle = load_cached(record["cache_path"])
            self.file_handles[subject] = handle
            xyzdir = np.asarray(handle["sensor_xyzdir"][:], dtype=np.float32)
            for row in xyzdir:
                if not np.all(np.isfinite(row[3:])) or np.linalg.norm(row[3:]) == 0:
                    norm = np.linalg.norm(row[:3])
                    row[3:] = row[:3] / norm if norm > 0 else 0
            record["sensor_xyzdir"] = xyzdir
            record["sensor_types"] = np.full(
                xyzdir.shape[0], self.eeg_sensor_type_id, dtype=np.int64
            )
            record["sfreq"] = float(handle.attrs["sample_freq"])
            record["n_samples"] = int(handle.attrs["n_samples"])
            record["duration"] = record["n_samples"] / record["sfreq"]

    def _make_recordings(self) -> List[Dict[str, Any]]:
        return [
            {
                "subject": subject,
                "session": "ses-001",
                "task": "alice_chapter_one",
                "duration": record["duration"],
                "segment_onsets": record["segment_onsets"],
            }
            for subject, record in sorted(self.subject_recordings.items())
        ]

    def _make_word_groups(self, recording: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        segment_onsets = recording["segment_onsets"]

        for event in self.word_events.itertuples(index=False):
            segment_idx = int(event.segment) - 1
            absolute_onset = float(segment_onsets[segment_idx]) + float(event.onset)
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
                    "segment": int(event.segment),
                    "order": int(event.order),
                    "subsegment_idx": len(current),
                }
            )
            if len(current) == self.words_per_segment:
                groups.append(current.copy())
                current = []
        return groups

    def __len__(self) -> int:
        return len(self.segment_index)

    def get_segment_words(self, idx: int) -> List[str]:
        recording_idx, group_idx = self.segment_index[idx]
        return [item["word"] for item in self.word_groups[recording_idx][group_idx]]

    def get_split_group(self, idx: int, group_kind: str) -> str:
        recording_idx, _ = self.segment_index[idx]
        recording = self.recordings[recording_idx]
        if group_kind == "sentence":
            return " ".join(self.get_segment_words(idx))
        if group_kind == "subject":
            return f"{self.dataset_name}:{recording['subject']}"
        if group_kind == "session":
            return f"{self.dataset_name}:{recording['subject']}:{recording['session']}"
        return (
            f"{self.dataset_name}:{recording['subject']}:"
            f"{recording['session']}:{recording['task']}"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        recording_idx, group_idx = self.segment_index[idx]
        recording = self.recordings[recording_idx]
        subject_record = self.subject_recordings[recording["subject"]]
        handle = self.file_handles[recording["subject"]]
        sfreq = subject_record["sfreq"]
        sensor_types = subject_record["sensor_types"].copy()
        word_group = self.word_groups[recording_idx][group_idx]
        window_samples = int(round(self.subsegment_duration * sfreq))

        subsegments = []
        for word in word_group:
            start = int(round(word["window_start"] * sfreq))
            chunk = np.asarray(
                handle["data"][:, start : start + window_samples], dtype=np.float32
            )
            if chunk.shape[1] != window_samples:
                raise RuntimeError("Unexpected truncated Alice EEG word window.")
            chunk = _process_single_chunk(
                chunk,
                sensor_types,
                sfreq,
                self.baseline_duration,
                self.clip_range,
            )
            subsegments.append(chunk.astype(np.float32, copy=False))

        eeg = np.concatenate(subsegments, axis=1)
        xyzdir = norm_sensor_positions(
            subject_record["sensor_xyzdir"].copy()
        ).astype(np.float32)
        original_channels = eeg.shape[0]
        if self.max_channel_dim is not None:
            if original_channels > self.max_channel_dim:
                raise ValueError(
                    f"Found {original_channels} channels but "
                    f"max_channel_dim={self.max_channel_dim}."
                )
            padding = self.max_channel_dim - original_channels
            eeg = np.pad(eeg, ((0, padding), (0, 0)))
            xyzdir = np.pad(xyzdir, ((0, padding), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, padding))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_channels] = 1
        else:
            sensor_mask = np.ones(original_channels, dtype=np.float32)

        boundaries = []
        cursor = 0
        for subsegment in subsegments:
            boundaries.append(
                {
                    "start_sample": cursor,
                    "end_sample": cursor + subsegment.shape[1],
                }
            )
            cursor += subsegment.shape[1]

        return {
            "meg": torch.from_numpy(eeg).float(),
            "subject": recording["subject"],
            "session": recording["session"],
            "task": recording["task"],
            "sensor_xyzdir": torch.from_numpy(xyzdir).float(),
            "sensor_types": torch.from_numpy(sensor_types).int(),
            "sensor_mask": torch.from_numpy(sensor_mask).float(),
            "words": [item["word"] for item in word_group],
            "subsegment_boundaries": boundaries,
            "recording_idx": recording_idx,
            "segment_idx": group_idx,
            "start_time": float(word_group[0]["window_start"]),
            "end_time": float(word_group[-1]["window_end"]),
        }

    def close(self) -> None:
        for handle in getattr(self, "file_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self.file_handles = {}

    def __del__(self):
        self.close()


if __name__ == "__main__":
    dataset = AliceEEGWordAlignedDataset(
        data_root="./datasets/alice_eeg",
        cache_dir="./data/cache/alice_eeg_word_aligned",
        max_channel_dim=64,
        subject_selection="main",
    )
    print(f"Dataset: {len(dataset)} segments")
    print(f"First EEG shape: {dataset[0]['meg'].shape}")
    dataset.close()
