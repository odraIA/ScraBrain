"""MEG-XL-compatible word-aligned loader for Weissbart listening EEG."""

from __future__ import annotations

import re
import warnings
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io as sio
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


class WeissbartEEGWordAlignedDataset(Dataset):
    """Expose Weissbart EEG with the same sample contract as MEG-XL datasets.

    Each item contains consecutive word-aligned EEG windows. Every window is
    independently baseline-corrected, robust-scaled and clipped with the same
    helper used by the Armeni and LibriBrain loaders, then concatenated in time.
    EEG is returned under ``meg`` because the CrissCross fine-tuning pipeline
    expects that key.
    """

    STORY_NAMES = (
        "AUNP01", "AUNP02", "AUNP03", "AUNP04", "AUNP05", "AUNP06",
        "AUNP07", "AUNP08", "BROP01", "BROP02", "BROP03", "FLOP01",
        "FLOP02", "FLOP03", "FLOP04",
    )
    PARTICIPANT_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14)
    _PARTICIPANT_RE = re.compile(r"P(?P<id>\d{1,2})", re.IGNORECASE)

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
        dataset_name: str = "weissbart_eeg",
        task_mode: str = "listening",
        tokenizer_name: str = "biocodec",
    ):
        self.data_root = self._resolve_root(Path(data_root))
        self.eeg_root = self.data_root / "eeg"
        self.stim_root = self.data_root / "stim"
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

        expected = self.words_per_segment * self.subsegment_duration
        if not np.isclose(self.segment_length, expected):
            raise ValueError(
                "segment_length must equal words_per_segment * subsegment_duration "
                f"({self.segment_length} != {expected})."
            )
        if self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive.")

        self.subjects = (
            {self._normalise_subject(value) for value in subjects}
            if subjects is not None else None
        )
        self.sessions = (
            {self._normalise_session(value) for value in sessions}
            if sessions is not None else None
        )
        self.tasks = (
            {str(value).strip().upper() for value in tasks}
            if tasks is not None else None
        )
        if self.sessions is not None and "ses-001" not in self.sessions:
            raise ValueError("Weissbart only exposes the logical session 'ses-001'.")

        self.onsets = self._load_onsets()
        self.annotation_paths = self._find_annotations()
        self.story_events = {
            story: self._read_word_events(path)
            for story, path in self.annotation_paths.items()
        }
        self.story_durations = {
            story: self._read_story_duration(story)
            for story in self.annotation_paths
        }

        self.subject_recordings = self._find_subject_recordings()
        if not self.subject_recordings:
            raise ValueError(f"No Weissbart BrainVision files found under {self.eeg_root}.")

        self._preprocess_subjects()
        self.file_handles: Dict[str, h5py.File] = {}
        self._open_subject_caches()
        self.recordings = self._make_story_recordings()
        self.word_groups = [self._make_word_groups(recording) for recording in self.recordings]
        self.segment_index = [
            (recording_idx, group_idx)
            for recording_idx, groups in enumerate(self.word_groups)
            for group_idx in range(len(groups))
        ]
        for index, (recording, groups) in enumerate(zip(self.recordings, self.word_groups)):
            print(
                f"Recording {index} ({recording['subject']} {recording['story']}): "
                f"Found {len(groups)} word-aligned segments"
            )
        if not self.segment_index:
            self.close()
            raise ValueError(
                "No complete Weissbart word-aligned segments were created. "
                "Check the timed CSV files or reduce words_per_segment."
            )

    @staticmethod
    def _resolve_root(path: Path) -> Path:
        for candidate in (path, path / "WeissbartSurprisal", path / "WeissbartEEG"):
            if (candidate / "eeg").exists() and (candidate / "stim").exists():
                return candidate
        raise FileNotFoundError(
            f"Expected Weissbart 'eeg' and 'stim' directories under {path}. "
            "Use the original release containing BrainVision EEG, onsets.mat "
            "and timed word annotations."
        )

    @staticmethod
    def _sensor_type_id(value: str) -> int:
        aliases = {
            "grad": 0, "gradiometer": 0, "mag": 1, "meg": 1,
            "magnetometer": 1, "eeg": 2,
        }
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown eeg_sensor_type {value!r}; choose {sorted(aliases)}.")
        return aliases[key]

    @classmethod
    def _normalise_subject(cls, value: str) -> str:
        text = re.sub(r"^sub-", "", str(value).strip(), flags=re.IGNORECASE)
        match = cls._PARTICIPANT_RE.search(text)
        if match:
            return f"P{int(match.group('id')):02d}"
        return f"P{int(text):02d}" if text.isdigit() else text.upper()

    @staticmethod
    def _normalise_session(value: str) -> str:
        text = str(value).strip().lower()
        if text in {"1", "001", "ses-1", "ses-001", "session-1", "session-001"}:
            return "ses-001"
        return text

    def _load_onsets(self) -> np.ndarray:
        candidates = [
            self.stim_root / "onsets.mat",
            self.data_root / "story_parts" / "onsets.mat",
        ]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            raise FileNotFoundError("Could not find stim/onsets.mat.")
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        if "onsets" not in mat:
            raise KeyError(f"Variable 'onsets' is missing from {path}.")
        onsets = np.asarray(mat["onsets"], dtype=np.float64)
        if onsets.ndim == 1:
            onsets = onsets[None, :]
        expected = (len(self.PARTICIPANT_IDS), len(self.STORY_NAMES))
        if onsets.shape == expected[::-1]:
            onsets = onsets.T
        if onsets.shape[0] < expected[0] or onsets.shape[1] < expected[1]:
            raise ValueError(f"Unexpected onsets shape {onsets.shape}; expected at least {expected}.")
        return onsets[: expected[0], : expected[1]]

    def _find_annotations(self) -> Dict[str, Path]:
        roots = [
            self.stim_root / "word_frequencies",
            self.data_root / "story_parts" / "word_frequencies",
            self.stim_root,
        ]
        requested = self.tasks or set(self.STORY_NAMES)
        result: Dict[str, Path] = {}
        for story in self.STORY_NAMES:
            if story not in requested:
                continue
            candidates: List[Path] = []
            for root in roots:
                if root.exists():
                    candidates.extend([
                        root / f"{story}_word_freq_timed.csv",
                        root / f"{story}_timed.csv",
                    ])
                    candidates.extend(sorted(root.glob(f"{story}*timed*.csv")))
            candidates.extend(sorted(self.stim_root.rglob(f"{story}*timed*.csv")))
            path = next((item for item in candidates if item.exists()), None)
            if path is None:
                warnings.warn(f"Timed annotation missing for {story}; skipping it.")
            else:
                result[story] = path
        if not result:
            raise FileNotFoundError(
                "No timed word CSVs found (expected e.g. "
                "stim/word_frequencies/AUNP01_word_freq_timed.csv)."
            )
        return result

    @staticmethod
    def _clean_word(value: Any) -> str:
        word = str(value).strip().strip('"').strip("'").lower()
        return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)

    def _read_word_events(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if not {"word", "onset"}.issubset(frame.columns):
            raise ValueError(
                f"{path} needs 'word' and 'onset' columns; found {list(frame.columns)}."
            )
        events = pd.DataFrame({
            "word": [self._clean_word(value) for value in frame["word"]],
            "onset": pd.to_numeric(frame["onset"], errors="coerce"),
        }).dropna(subset=["onset"])
        return (
            events[~events["word"].isin({"", "sp", "silence", "s"})]
            .sort_values("onset")
            .reset_index(drop=True)
        )

    def _read_story_duration(self, story: str) -> Optional[float]:
        candidates = [
            self.stim_root / "alignment_data" / story / f"{story}.wav",
            self.stim_root / "alignement_data" / story / f"{story}.wav",
            self.data_root / "story_parts" / "alignment_data" / story / f"{story}.wav",
            self.data_root / "story_parts" / "alignement_data" / story / f"{story}.wav",
        ]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            warnings.warn(f"Audio missing for {story}; checking only EEG bounds.")
            return None
        try:
            with wave.open(str(path), "rb") as stream:
                return stream.getnframes() / float(stream.getframerate())
        except (wave.Error, EOFError):
            warnings.warn(f"Could not read audio duration from {path}.")
            return None

    def _participant_id(self, path: Path) -> Optional[int]:
        match = self._PARTICIPANT_RE.search(str(path))
        return int(match.group("id")) if match else None

    def _find_subject_recordings(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for raw_path in sorted(self.eeg_root.rglob("*.vhdr")):
            participant_id = self._participant_id(raw_path)
            if participant_id not in self.PARTICIPANT_IDS:
                continue
            subject = f"P{participant_id:02d}"
            if self.subjects is not None and subject not in self.subjects:
                continue
            if subject in result:
                warnings.warn(f"Ignoring duplicate VHDR for {subject}: {raw_path}")
                continue
            filter_name = "EEG_only"
            if self.channel_filter is not None:
                filter_name += f"_{getattr(self.channel_filter, '__name__', 'custom')}"
            filter_name += f"_type{self.eeg_sensor_type_id}"
            result[subject] = {
                "subject": subject,
                "session": "ses-001",
                "participant_idx": self.PARTICIPANT_IDS.index(participant_id),
                "raw_path": raw_path,
                "filter_name": filter_name,
                "cache_path": get_cache_path(
                    self.cache_dir, subject, "ses-001", "continuous",
                    l_freq=self.l_freq, h_freq=self.h_freq,
                    target_sfreq=self.target_sfreq,
                    channel_filter_name=filter_name,
                    dataset_name=self.dataset_name, task_mode=self.task_mode,
                    segment_length=self.segment_length,
                    subsegment_duration=self.subsegment_duration,
                    window_onset_offset=self.window_onset_offset,
                    tokenizer_name=self.tokenizer_name,
                ),
            }
        return result

    def _preprocess_subjects(self) -> None:
        total = len(self.subject_recordings)
        for index, record in enumerate(self.subject_recordings.values(), 1):
            if is_hdf5_cache_readable(record["cache_path"]):
                print(f"Using cached Weissbart recording {index}/{total}: {record['subject']}")
                continue
            print(f"Preprocessing Weissbart recording {index}/{total}: {record['subject']}")
            raw = mne.io.read_raw_brainvision(str(record["raw_path"]), preload=True, verbose=False)
            picks = mne.pick_types(
                raw.info, meg=False, eeg=True, eog=False, ecg=False, emg=False,
                stim=False, misc=False, exclude=[],
            )
            if len(picks) == 0:
                raise ValueError(f"No EEG channels found in {record['raw_path']}.")
            raw.pick(picks)
            if self.channel_filter is not None:
                keep = [name for name in raw.ch_names if self.channel_filter(name)]
                if not keep:
                    raise ValueError("channel_filter removed every Weissbart EEG channel.")
                raw.pick(keep)
            try:
                raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
            except (TypeError, ValueError):
                pass
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
                    "task": "continuous",
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

    def _make_story_recordings(self) -> List[Dict[str, Any]]:
        result = []
        for subject, subject_record in sorted(self.subject_recordings.items()):
            participant_idx = subject_record["participant_idx"]
            for story_idx, story in enumerate(self.STORY_NAMES):
                if story not in self.annotation_paths:
                    continue
                onset = float(self.onsets[participant_idx, story_idx])
                if not np.isfinite(onset) or onset < 0 or onset >= subject_record["duration"]:
                    continue
                result.append({
                    "subject": subject,
                    "session": "ses-001",
                    "task": story,
                    "story": story,
                    "story_onset": onset,
                    "story_duration": self.story_durations.get(story),
                    "duration": subject_record["duration"],
                })
        return result

    def _make_word_groups(self, recording: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for event in self.story_events[recording["story"]].itertuples(index=False):
            relative_onset = float(event.onset)
            relative_start = relative_onset + self.window_onset_offset
            relative_end = relative_start + self.subsegment_duration
            start = recording["story_onset"] + relative_start
            end = start + self.subsegment_duration
            valid = relative_start >= 0 and start >= 0 and end <= recording["duration"]
            if recording["story_duration"] is not None:
                valid = valid and relative_end <= recording["story_duration"]
            if not valid:
                current = []
                continue
            current.append({
                "word": str(event.word),
                "onset": recording["story_onset"] + relative_onset,
                "window_start": start,
                "window_end": end,
                "subsegment_idx": len(current),
            })
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
            f"{recording['session']}:{recording['story']}"
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
                handle["data"][:, start:start + window_samples], dtype=np.float32
            )
            if chunk.shape[1] != window_samples:
                raise RuntimeError("Unexpected truncated Weissbart EEG word window.")
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
            boundaries.append({
                "start_sample": cursor,
                "end_sample": cursor + subsegment.shape[1],
            })
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
    dataset = WeissbartEEGWordAlignedDataset(
        data_root="./datasets/WeissbartEEG",
        cache_dir="./data/cache/weissbart_eeg_word_aligned",
        max_channel_dim=64,
    )
    print(f"Dataset: {len(dataset)} segments")
    print(f"First EEG shape: {dataset[0]['meg'].shape}")
    dataset.close()
