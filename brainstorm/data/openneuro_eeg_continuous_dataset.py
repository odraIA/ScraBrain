"""Continuous OpenNeuro EEG dataset for MEG-XL-style self-supervised training.

The original MEG-XL pre-training pipeline consumes fixed-duration neural-signal
windows. It does not need words, transcripts, or language-specific tokenization.

For normal ``task-listening`` recordings this dataset uses the complete EEG
recording. For ds007808 ``task-listeningcovert`` recordings, the optional
``listening_only`` policy extracts every interval whose ``trial_type`` is
``listening``, concatenates those intervals into a virtual listening-only
stream, and creates fixed-length windows from that stream.

To guarantee that 100% of the selected listening samples are represented while
keeping a fixed segment length, the last window overlaps the previous one when
there is a remainder. Very short streams can be repeated instead of padded.
No ``covert`` interval is included in a listening-only stream.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_segment_with_subsegments


EEG_SENSOR_TYPE_ID = 2
_RAW_SUFFIXES = {".edf", ".bdf", ".vhdr"}


@dataclass(frozen=True)
class IntervalRef:
    """A real EEG interval inside one cached source recording."""

    source_idx: int
    start_sample: int
    end_sample: int

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample


def _entity_dict(path: Path) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    for part in path.name.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        entities[key] = value.split(".")[0]
    return entities


def _normalize_filter(values: Optional[Sequence[str]], prefix: str) -> Optional[set[str]]:
    if values is None:
        return None
    normalized = set()
    for value in values:
        text = str(value)
        if text.startswith(prefix):
            text = text[len(prefix) :]
        normalized.add(text.lower())
    return normalized


def _normalize_sensor_xyzdir(sensor_xyzdir: np.ndarray) -> np.ndarray:
    sensor_xyzdir = np.nan_to_num(sensor_xyzdir.astype(np.float32, copy=True), copy=False)
    positions = sensor_xyzdir[:, :3]
    if positions.size == 0:
        return sensor_xyzdir

    centered = positions - np.mean(positions, axis=0, keepdims=True)
    scale = float(np.sqrt(3.0 * np.mean(np.sum(centered**2, axis=1))))
    if not np.isfinite(scale) or scale <= 0:
        sensor_xyzdir[:, :3] = 0.0
    else:
        sensor_xyzdir[:, :3] = centered / scale
    return sensor_xyzdir


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def merge_intervals(
    intervals: Sequence[Tuple[int, int]],
    max_gap_samples: int = 0,
) -> List[Tuple[int, int]]:
    """Merge overlapping or nearly adjacent sample intervals."""

    cleaned = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    if not cleaned:
        return []

    merged: List[Tuple[int, int]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + max_gap_samples:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def segment_starts_cover_all(total_samples: int, segment_samples: int) -> List[int]:
    """Return fixed-length starts that cover every sample at least once.

    Non-overlapping windows are created first. If a remainder exists, one final
    window ending exactly at ``total_samples`` is added, which overlaps the
    preceding window instead of discarding the remainder.
    """

    total_samples = int(total_samples)
    segment_samples = int(segment_samples)
    if total_samples <= 0:
        return []
    if segment_samples <= 0:
        raise ValueError(f"segment_samples must be positive, got {segment_samples}")
    if total_samples <= segment_samples:
        return [0]

    last_start = total_samples - segment_samples
    starts = list(range(0, last_start + 1, segment_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


class OpenNeuroEEGContinuousDataset(Dataset):
    """BIDS-like EEG dataset returning fixed continuous windows under ``meg``.

    Parameters specific to ds007808
    --------------------------------
    listeningcovert_policy:
        ``"listening_only"`` keeps only rows with ``trial_type=listening`` from
        ``task-listeningcovert``. ``"full_recording"`` uses the complete file.
    group_listeningcovert_by:
        How listening-only intervals are combined before segmentation. The
        default ``"subject_session"`` combines all runs from the same subject
        and session, which usually produces enough listening material for a
        150-second context window.
    listening_interval_start:
        ``"onset"`` uses the full BIDS event interval, preserving 100% of the
        trial labelled listening. ``"wav_onset"`` starts at actual audio onset
        when that column is available and ends at ``onset + duration``.
    cover_all_samples:
        If true, an overlapping final window is added so no selected sample is
        discarded.
    short_stream_policy:
        If a listening-only stream is shorter than one segment, ``"repeat"``
        repeats real listening samples cyclically. ``"zero_pad"`` pads with
        zeros; ``"error"`` raises an exception.
    """

    raw_suffixes = _RAW_SUFFIXES

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        cache_dir: str = "./data/cache/eeg_continuous",
        subjects: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[str]] = None,
        tasks: Optional[Sequence[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter: Optional[Callable[[str], bool]] = None,
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
        listeningcovert_policy: str = "listening_only",
        listening_trial_type: str = "listening",
        listening_interval_start: str = "onset",
        group_listeningcovert_by: str = "subject_session",
        cover_all_samples: bool = True,
        short_stream_policy: str = "repeat",
        merge_gap_seconds: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_name = str(dataset_name)
        self.segment_length = float(segment_length)
        self.subsegment_duration = float(subsegment_duration)
        self.cache_dir = Path(cache_dir) / self.dataset_name / "continuous"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.target_sfreq = float(target_sfreq)
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.baseline_duration = float(baseline_duration)
        self.clip_range = tuple(float(value) for value in clip_range)

        self.listeningcovert_policy = str(listeningcovert_policy).lower()
        self.listening_trial_type = str(listening_trial_type).strip().lower()
        self.listening_interval_start = str(listening_interval_start).lower()
        self.group_listeningcovert_by = str(group_listeningcovert_by).lower()
        self.cover_all_samples = bool(cover_all_samples)
        self.short_stream_policy = str(short_stream_policy).lower()
        self.merge_gap_seconds = float(merge_gap_seconds)

        self.subjects = _normalize_filter(subjects, "sub-")
        self.sessions = _normalize_filter(sessions, "ses-")
        self.tasks = {str(task).lower() for task in tasks} if tasks is not None else None

        self._validate_configuration()
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")

        self.source_recordings = self._discover_source_recordings()
        if not self.source_recordings:
            raise ValueError(
                f"No EEG recordings found in {self.data_root} for subjects={subjects}, "
                f"sessions={sessions}, tasks={tasks}"
            )

        self._preprocess_all()
        self.recordings = self._build_virtual_streams()
        self.segment_starts: List[List[int]] = []
        self.segment_index = self._build_segment_index()
        self._file_handles: Dict[int, h5py.File] = {}

        if not self.segment_index:
            raise ValueError(
                f"No continuous EEG segments found for {self.dataset_name}. "
                "Check task filters and listening event metadata."
            )

        self.coverage_report = self._build_coverage_report()
        self._print_summary()

    def _validate_configuration(self) -> None:
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        if self.subsegment_duration <= 0:
            raise ValueError("subsegment_duration must be positive")
        if self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive")
        if self.listeningcovert_policy not in {"listening_only", "full_recording"}:
            raise ValueError(
                "listeningcovert_policy must be 'listening_only' or 'full_recording'"
            )
        if self.listening_interval_start not in {"onset", "wav_onset"}:
            raise ValueError("listening_interval_start must be 'onset' or 'wav_onset'")
        if self.group_listeningcovert_by not in {
            "recording",
            "subject_session",
            "subject",
        }:
            raise ValueError(
                "group_listeningcovert_by must be recording, subject_session, or subject"
            )
        if self.short_stream_policy not in {"repeat", "zero_pad", "error"}:
            raise ValueError("short_stream_policy must be repeat, zero_pad, or error")

    def _discover_source_recordings(self) -> List[Dict[str, Any]]:
        recordings: List[Dict[str, Any]] = []

        for raw_path in sorted(self.data_root.rglob("*_eeg.*")):
            if raw_path.suffix.lower() not in self.raw_suffixes:
                continue
            if not raw_path.is_file():
                continue

            entities = _entity_dict(raw_path)
            subject_id = entities.get("sub", "")
            session_id = entities.get("ses", "")
            task = entities.get("task", "").lower()
            run = entities.get("run", "")

            if self.subjects is not None and subject_id.lower() not in self.subjects:
                continue
            if self.sessions is not None and session_id.lower() not in self.sessions:
                continue
            if self.tasks is not None and task not in self.tasks:
                continue
            if task == "speechopen":
                continue

            subject = f"sub-{subject_id}" if subject_id else "sub-unknown"
            session = f"ses-{session_id}" if session_id else ""
            base = raw_path.name.rsplit("_eeg.", 1)[0]
            events_path = raw_path.with_name(f"{base}_events.tsv")

            recording = {
                "raw_path": raw_path,
                "events_path": events_path if events_path.exists() else None,
                "subject": subject,
                "session": session,
                "task": task or "unknown",
                "run": run,
                "entities": entities,
            }
            recording["cache_path"] = self._cache_path(recording)
            recordings.append(recording)

        return recordings

    def _cache_path(self, recording: Dict[str, Any]) -> Path:
        raw_path = Path(recording["raw_path"])
        try:
            stat = raw_path.stat()
            file_identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            file_identity = {"size": None, "mtime_ns": None}

        channel_filter_name = None
        if self.channel_filter is not None:
            channel_filter_name = getattr(self.channel_filter, "__name__", repr(self.channel_filter))

        key = {
            "raw_path": str(raw_path.resolve()),
            "file_identity": file_identity,
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "target_sfreq": self.target_sfreq,
            "channel_filter": channel_filter_name,
        }
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        safe_stem = raw_path.name.replace(".", "_")
        return self.cache_dir / f"{safe_stem}_{digest}.h5"

    @staticmethod
    def _read_raw(raw_path: Path, preload: bool = True) -> mne.io.BaseRaw:
        suffix = raw_path.suffix.lower()
        if suffix == ".edf":
            return mne.io.read_raw_edf(raw_path, preload=preload, verbose=False)
        if suffix == ".bdf":
            return mne.io.read_raw_bdf(raw_path, preload=preload, verbose=False)
        if suffix == ".vhdr":
            return mne.io.read_raw_brainvision(raw_path, preload=preload, verbose=False)
        raise ValueError(f"Unsupported EEG format: {raw_path}")

    @staticmethod
    def _cache_is_readable(path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False
        try:
            with h5py.File(path, "r") as h5_file:
                required = {"data", "sensor_xyzdir", "sensor_types", "channel_names"}
                return required.issubset(h5_file.keys()) and int(h5_file.attrs["n_samples"]) > 0
        except Exception:
            return False

    def _apply_eeg_montage(
        self,
        raw: mne.io.BaseRaw,
        raw_path: Path,
    ) -> None:
        for index, recording in enumerate(self.source_recordings, start=1):
            cache_path = Path(recording["cache_path"])
            if self._cache_is_readable(cache_path):
                print(
                    f"Using cached EEG {index}/{len(self.source_recordings)}: "
                    f"{recording['subject']} {recording['session']} "
                    f"{recording['task']} run-{recording['run']}"
                )
                continue

            print(
                f"Preprocessing EEG {index}/{len(self.source_recordings)}: "
                f"{recording['raw_path']}"
            )
            self._preprocess_recording(recording)

    def _apply_eeg_montage(raw, raw_path: Path) -> None:
        channel_set = set(raw.ch_names)

        # ---------------------------------------------------------
        # ds004408: BioSemi 128
        # ---------------------------------------------------------
        biosemi128_channels = {
            f"{group}{index}"
            for group in ("A", "B", "C", "D")
            for index in range(1, 33)
        }

        if channel_set == biosemi128_channels:
            montage = mne.channels.make_standard_montage("biosemi128")
            raw.set_montage(
                montage,
                match_case=True,
                on_missing="raise",
            )
            print("Applied BioSemi 128 montage")
            return

        # ---------------------------------------------------------
        # ds007808: g.Pangolin coordinates from BIDS sidecars
        # ---------------------------------------------------------
        if all(re.fullmatch(r"EEG\d{3}", name) for name in raw.ch_names):
            eeg_dir = raw_path.parent

            acquisition_match = re.search(r"_acq-([^_]+)", raw_path.name)
            acquisition = (
                acquisition_match.group(1)
                if acquisition_match is not None
                else None
            )

            electrode_candidates = list(eeg_dir.glob("*_electrodes.tsv"))

            if acquisition is not None:
                acquisition_candidates = [
                    path
                    for path in electrode_candidates
                    if f"_acq-{acquisition}" in path.name
                ]

                if acquisition_candidates:
                    electrode_candidates = acquisition_candidates

            if len(electrode_candidates) != 1:
                raise RuntimeError(
                    f"Could not identify a unique electrodes.tsv for {raw_path}. "
                    f"Candidates: {electrode_candidates}"
                )

            electrodes_path = electrode_candidates[0]
            coordsystem_path = electrodes_path.with_name(
                electrodes_path.name.replace(
                    "_electrodes.tsv",
                    "_coordsystem.json",
                )
            )

            if not coordsystem_path.exists():
                raise FileNotFoundError(
                    f"Missing coordinate system file: {coordsystem_path}"
                )

            electrodes = pd.read_csv(electrodes_path, sep="\t")

            required_columns = {"name", "x", "y", "z"}
            missing_columns = required_columns - set(electrodes.columns)

            if missing_columns:
                raise ValueError(
                    f"{electrodes_path} is missing columns "
                    f"{sorted(missing_columns)}"
                )

            with open(coordsystem_path, "r", encoding="utf-8") as file:
                coordinate_metadata = json.load(file)

            unit = str(
                coordinate_metadata.get("EEGCoordinateUnits", "m")
            ).lower()

            unit_scale = {
                "m": 1.0,
                "cm": 1e-2,
                "mm": 1e-3,
            }

            if unit not in unit_scale:
                raise ValueError(
                    f"Unsupported EEG coordinate unit: {unit}"
                )

            scale = unit_scale[unit]
            raw_channels = set(raw.ch_names)
            channel_positions = {}

            for _, row in electrodes.iterrows():
                name = str(row["name"]).strip()

                if name not in raw_channels:
                    continue

                xyz = np.asarray(
                    [row["x"], row["y"], row["z"]],
                    dtype=np.float64,
                )

                if not np.all(np.isfinite(xyz)):
                    continue

                channel_positions[name] = xyz * scale

            missing_positions = raw_channels - set(channel_positions)

            if missing_positions:
                raise ValueError(
                    f"Missing positions for {len(missing_positions)} channels: "
                    f"{sorted(missing_positions)[:10]}"
                )

            montage = mne.channels.make_dig_montage(
                ch_pos=channel_positions,
                coord_frame="head",
            )

            raw.set_montage(
                montage,
                match_case=True,
                on_missing="raise",
            )

            print(
                f"Applied g.Pangolin BIDS montage from "
                f"{electrodes_path.name}"
            )
            return

        print(
            "No known montage detected; existing channel positions "
            "will be used."
        )

    def _preprocess_recording(self, recording: Dict[str, Any]) -> None:
        raw_path = Path(recording["raw_path"])
        raw = self._read_raw(raw_path, preload=True)

        try:
            # ---------------------------------------------------------
            # Apply the channel types declared in the BIDS channels.tsv
            # ---------------------------------------------------------
            channels_path_value = recording.get("channels_path")

            if channels_path_value is not None:
                channels_path = Path(channels_path_value)
            else:
                marker = "_eeg."

                if marker not in raw_path.name:
                    raise ValueError(
                        f"Cannot infer channels.tsv path from {raw_path}"
                    )

                bids_prefix = raw_path.name.split(marker, maxsplit=1)[0]
                channels_path = raw_path.with_name(
                    f"{bids_prefix}_channels.tsv"
                )

            if not channels_path.exists():
                raise FileNotFoundError(
                    f"Missing BIDS channels file for {raw_path}: "
                    f"{channels_path}"
                )

            channels_df = pd.read_csv(channels_path, sep="\t")

            required_columns = {"name", "type"}
            missing_columns = required_columns - set(channels_df.columns)

            if missing_columns:
                raise ValueError(
                    f"{channels_path} is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            bids_to_mne_type = {
                "EEG": "eeg",
                "EOG": "eog",
                "ECG": "ecg",
                "EMG": "emg",
                "MISC": "misc",
                "TRIG": "stim",
                "STIM": "stim",
            }

            channel_types = {}
            missing_from_raw = []

            for _, row in channels_df.iterrows():
                channel_name = str(row["name"]).strip()
                bids_type = str(row["type"]).strip().upper()

                if channel_name not in raw.ch_names:
                    missing_from_raw.append(channel_name)
                    continue

                channel_types[channel_name] = bids_to_mne_type.get(
                    bids_type,
                    "misc",
                )

            if not channel_types:
                raise ValueError(
                    f"No channels from {channels_path} matched "
                    f"the channels in {raw_path}"
                )

            raw.set_channel_types(
                channel_types,
                on_unit_change="ignore",
            )

            if missing_from_raw:
                warnings.warn(
                    f"{len(missing_from_raw)} channels from "
                    f"{channels_path} were not found in {raw_path}: "
                    f"{missing_from_raw[:10]}"
                )

            # ---------------------------------------------------------
            # Keep only genuine EEG channels
            # ---------------------------------------------------------
            eeg_picks = mne.pick_types(
                raw.info,
                eeg=True,
                eog=False,
                ecg=False,
                emg=False,
                misc=False,
                stim=False,
                exclude=[],
            )

            if len(eeg_picks) == 0:
                raise ValueError(
                    f"No EEG channels found in {raw_path}"
                )

            channel_names = [
                raw.ch_names[index]
                for index in eeg_picks
            ]

            if self.channel_filter is not None:
                channel_names = [
                    name
                    for name in channel_names
                    if self.channel_filter(name)
                ]

            if not channel_names:
                raise ValueError(
                    f"Channel filter removed every EEG channel "
                    f"in {raw_path}"
                )

            total_channels = len(raw.ch_names)
            raw.pick(channel_names)

            self._apply_eeg_montage(raw, raw_path)

            print(
                f"{raw_path.name}: selected "
                f"{len(raw.ch_names)} EEG channels from "
                f"{total_channels} total channels"
            )

            # ---------------------------------------------------------
            # Filtering and resampling
            # ---------------------------------------------------------
            nyquist = float(raw.info["sfreq"]) / 2.0

            if self.h_freq >= nyquist:
                raise ValueError(
                    f"h_freq={self.h_freq} must be below the "
                    f"original Nyquist frequency {nyquist:.3f} Hz "
                    f"for {raw_path}"
                )

            raw.filter(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                picks="eeg",
                n_jobs=1,
                verbose=False,
            )

            if not math.isclose(
                float(raw.info["sfreq"]),
                self.target_sfreq,
                rel_tol=0,
                abs_tol=1e-6,
            ):
                raw.resample(
                    self.target_sfreq,
                    n_jobs=1,
                    verbose=False,
                )

            # ---------------------------------------------------------
            # Cache preprocessed EEG
            # ---------------------------------------------------------
            data = raw.get_data().astype(
                np.float32,
                copy=False,
            )

            sensor_xyzdir = self._sensor_xyzdir(raw)

            sensor_types = np.full(
                data.shape[0],
                EEG_SENSOR_TYPE_ID,
                dtype=np.int16,
            )

            channel_names_array = np.asarray(
                raw.ch_names,
                dtype=h5py.string_dtype("utf-8"),
            )

            cache_path = Path(recording["cache_path"])
            cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path = cache_path.with_suffix(
                cache_path.suffix + ".tmp"
            )

            if temporary_path.exists():
                temporary_path.unlink()

            with h5py.File(temporary_path, "w") as h5_file:
                h5_file.create_dataset(
                    "data",
                    data=data,
                    compression="lzf",
                )

                h5_file.create_dataset(
                    "sensor_xyzdir",
                    data=sensor_xyzdir,
                )

                h5_file.create_dataset(
                    "sensor_types",
                    data=sensor_types,
                )

                h5_file.create_dataset(
                    "channel_names",
                    data=channel_names_array,
                )

                h5_file.attrs["n_samples"] = int(data.shape[1])
                h5_file.attrs["n_channels"] = int(data.shape[0])
                h5_file.attrs["sample_freq"] = float(
                    self.target_sfreq
                )
                h5_file.attrs["subject"] = recording["subject"]
                h5_file.attrs["session"] = recording["session"]
                h5_file.attrs["task"] = recording["task"]
                h5_file.attrs["run"] = recording["run"]
                h5_file.attrs["raw_path"] = str(raw_path)
                h5_file.attrs["channels_path"] = str(
                    channels_path
                )
                h5_file.attrs["l_freq"] = self.l_freq
                h5_file.attrs["h_freq"] = self.h_freq

            temporary_path.replace(cache_path)

        finally:
            close = getattr(raw, "close", None)

            if callable(close):
                close()

    @staticmethod
    def _sensor_xyzdir(raw: mne.io.BaseRaw) -> np.ndarray:
        rows: List[np.ndarray] = []
        missing_positions = 0
        for channel in raw.info["chs"]:
            position = np.asarray(channel["loc"][:3], dtype=np.float32)
            if not np.all(np.isfinite(position)) or np.linalg.norm(position) <= 0:
                position = np.zeros(3, dtype=np.float32)
                direction = np.zeros(3, dtype=np.float32)
                missing_positions += 1
            else:
                direction = position / np.linalg.norm(position)
            rows.append(np.concatenate([position, direction]).astype(np.float32))

        if missing_positions:
            warnings.warn(
                f"{missing_positions}/{len(rows)} EEG channels have no usable 3D position; "
                "their position and orientation embeddings will be zero.",
                RuntimeWarning,
            )
        return _normalize_sensor_xyzdir(np.stack(rows, axis=0))

    def _source_metadata(self, source_idx: int) -> Dict[str, Any]:
        recording = self.source_recordings[source_idx]
        with h5py.File(recording["cache_path"], "r") as h5_file:
            channel_names = tuple(
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in h5_file["channel_names"][:]
            )
            sensor_xyzdir = np.asarray(h5_file["sensor_xyzdir"][:], dtype=np.float32)
            signature_payload = "\n".join(channel_names).encode("utf-8")
            signature_payload += np.round(sensor_xyzdir, 3).tobytes()
            signature = hashlib.sha1(signature_payload).hexdigest()[:16]
            return {
                "n_samples": int(h5_file.attrs["n_samples"]),
                "sample_freq": float(h5_file.attrs["sample_freq"]),
                "channel_names": channel_names,
                "sensor_signature": signature,
            }

    def _listening_intervals(
        self,
        recording: Dict[str, Any],
        n_samples: int,
        sample_freq: float,
    ) -> List[Tuple[int, int]]:
        events_path = recording.get("events_path")
        if events_path is None or not Path(events_path).exists():
            raise FileNotFoundError(
                f"listening_only requires an events.tsv sidecar for {recording['raw_path']}"
            )

        events = pd.read_csv(events_path, sep="\t")
        required = {"onset", "duration", "trial_type"}
        missing = required.difference(events.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {events_path}")

        selected: List[Tuple[int, int]] = []
        for _, row in events.iterrows():
            trial_type = str(row.get("trial_type", "")).strip().lower()
            if trial_type != self.listening_trial_type:
                continue

            onset = _safe_float(row.get("onset"))
            duration = _safe_float(row.get("duration"))
            if onset is None or duration is None or duration <= 0:
                continue

            event_end = onset + duration
            start_seconds = onset
            if self.listening_interval_start == "wav_onset":
                wav_onset = _safe_float(row.get("wav_onset"))
                if wav_onset is not None:
                    start_seconds = wav_onset

            start_seconds = max(0.0, start_seconds)
            end_seconds = max(start_seconds, event_end)
            start_sample = max(0, min(n_samples, int(round(start_seconds * sample_freq))))
            end_sample = max(0, min(n_samples, int(round(end_seconds * sample_freq))))
            if end_sample > start_sample:
                selected.append((start_sample, end_sample))

        merged = merge_intervals(
            selected,
            max_gap_samples=max(0, int(round(self.merge_gap_seconds * sample_freq))),
        )
        if not merged:
            raise ValueError(
                f"No trial_type={self.listening_trial_type!r} intervals found in {events_path}"
            )
        return merged

    def _listening_group_key(
        self,
        recording: Dict[str, Any],
        source_idx: int,
        sensor_signature: str,
    ) -> Tuple[str, ...]:
        if self.group_listeningcovert_by == "recording":
            return ("recording", str(source_idx), sensor_signature)
        if self.group_listeningcovert_by == "subject":
            return (recording["subject"], recording["task"], sensor_signature)
        return (
            recording["subject"],
            recording["session"],
            recording["task"],
            sensor_signature,
        )

    def _build_virtual_streams(self) -> List[Dict[str, Any]]:
        streams: List[Dict[str, Any]] = []
        listening_groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        for source_idx, recording in enumerate(self.source_recordings):
            metadata = self._source_metadata(source_idx)
            n_samples = metadata["n_samples"]
            sample_freq = metadata["sample_freq"]
            signature = metadata["sensor_signature"]
            task = recording["task"].lower()

            if task == "listeningcovert" and self.listeningcovert_policy == "listening_only":
                intervals = self._listening_intervals(recording, n_samples, sample_freq)
                key = self._listening_group_key(recording, source_idx, signature)
                group = listening_groups.setdefault(
                    key,
                    {
                        "subject": recording["subject"],
                        "sessions": set(),
                        "task": recording["task"],
                        "runs": [],
                        "source_indices": [],
                        "intervals": [],
                        "sensor_signature": signature,
                        "content_mode": "listening_only",
                    },
                )
                group["sessions"].add(recording["session"])
                group["runs"].append(recording["run"])
                group["source_indices"].append(source_idx)
                group["intervals"].extend(
                    IntervalRef(source_idx, start, end) for start, end in intervals
                )
                continue

            streams.append(
                self._finalize_stream(
                    {
                        "subject": recording["subject"],
                        "sessions": {recording["session"]},
                        "task": recording["task"],
                        "runs": [recording["run"]],
                        "source_indices": [source_idx],
                        "intervals": [IntervalRef(source_idx, 0, n_samples)],
                        "sensor_signature": signature,
                        "content_mode": "full_recording",
                    }
                )
            )

        for key in sorted(listening_groups, key=str):
            streams.append(self._finalize_stream(listening_groups[key]))

        return streams

    @staticmethod
    def _finalize_stream(stream: Dict[str, Any]) -> Dict[str, Any]:
        intervals: List[IntervalRef] = list(stream["intervals"])
        cumulative_ends: List[int] = []
        total_samples = 0
        for interval in intervals:
            total_samples += interval.length
            cumulative_ends.append(total_samples)

        stream = dict(stream)
        stream["intervals"] = intervals
        stream["cumulative_ends"] = cumulative_ends
        stream["total_samples"] = total_samples
        stream["session"] = ",".join(sorted(session for session in stream["sessions"] if session))
        stream["run"] = ",".join(str(run) for run in stream["runs"] if str(run))
        stream["recording_name"] = (
            f"{stream['subject']}:{stream['session']}:{stream['task']}:"
            f"{stream['content_mode']}"
        )
        return stream

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        segment_samples = int(round(self.segment_length * self.target_sfreq))
        segment_index: List[Tuple[int, int]] = []

        for stream_idx, stream in enumerate(self.recordings):
            total_samples = int(stream["total_samples"])
            if total_samples <= 0:
                self.segment_starts.append([])
                continue

            if total_samples < segment_samples and self.short_stream_policy == "error":
                raise ValueError(
                    f"Listening stream {stream['recording_name']} has only "
                    f"{total_samples / self.target_sfreq:.2f}s, shorter than "
                    f"segment_length={self.segment_length}s"
                )

            if self.cover_all_samples:
                starts = segment_starts_cover_all(total_samples, segment_samples)
            else:
                starts = list(range(0, max(0, total_samples - segment_samples + 1), segment_samples))
                if total_samples < segment_samples and self.short_stream_policy != "error":
                    starts = [0]

            self.segment_starts.append(starts)
            for segment_idx in range(len(starts)):
                segment_index.append((stream_idx, segment_idx))

        return segment_index

    def _build_coverage_report(self) -> Dict[str, Any]:
        segment_samples = int(round(self.segment_length * self.target_sfreq))
        stream_reports = []
        total_selected = 0
        total_segment_samples = 0

        for stream_idx, stream in enumerate(self.recordings):
            selected = int(stream["total_samples"])
            segment_count = len(self.segment_starts[stream_idx])
            represented = segment_count * segment_samples
            total_selected += selected
            total_segment_samples += represented
            stream_reports.append(
                {
                    "recording_name": stream["recording_name"],
                    "content_mode": stream["content_mode"],
                    "selected_seconds": selected / self.target_sfreq,
                    "segments": segment_count,
                    "represented_seconds_with_overlap": represented / self.target_sfreq,
                    "all_selected_samples_covered": bool(segment_count) if selected else True,
                }
            )

        return {
            "dataset_name": self.dataset_name,
            "source_recordings": len(self.source_recordings),
            "virtual_streams": len(self.recordings),
            "segments": len(self.segment_index),
            "selected_seconds": total_selected / self.target_sfreq,
            "represented_seconds_with_overlap": total_segment_samples / self.target_sfreq,
            "streams": stream_reports,
        }

    def _print_summary(self) -> None:
        listening_only = [
            stream for stream in self.recordings if stream["content_mode"] == "listening_only"
        ]
        listening_seconds = sum(stream["total_samples"] for stream in listening_only) / self.target_sfreq
        print(
            f"{self.dataset_name}: {len(self.source_recordings)} source recordings -> "
            f"{len(self.recordings)} continuous streams -> {len(self.segment_index)} segments"
        )
        if listening_only:
            print(
                f"{self.dataset_name}: selected {listening_seconds / 3600.0:.3f} h from "
                f"trial_type={self.listening_trial_type!r} inside listeningcovert; "
                "covert intervals excluded"
            )

    def __len__(self) -> int:
        return len(self.segment_index)

    def _get_h5(self, source_idx: int) -> h5py.File:
        handle = self._file_handles.get(source_idx)
        if handle is None or not handle.id.valid:
            handle = h5py.File(self.source_recordings[source_idx]["cache_path"], "r")
            self._file_handles[source_idx] = handle
        return handle

    def _read_once(self, stream: Dict[str, Any], start: int, length: int) -> np.ndarray:
        if length <= 0:
            return np.empty((0, 0), dtype=np.float32)

        cumulative_ends: List[int] = stream["cumulative_ends"]
        intervals: List[IntervalRef] = stream["intervals"]
        interval_idx = bisect.bisect_right(cumulative_ends, start)
        chunks: List[np.ndarray] = []
        remaining = length
        virtual_position = start

        while remaining > 0 and interval_idx < len(intervals):
            interval = intervals[interval_idx]
            interval_virtual_start = 0 if interval_idx == 0 else cumulative_ends[interval_idx - 1]
            offset = max(0, virtual_position - interval_virtual_start)
            available = interval.length - offset
            take = min(remaining, available)
            if take <= 0:
                interval_idx += 1
                continue

            h5_file = self._get_h5(interval.source_idx)
            real_start = interval.start_sample + offset
            real_end = real_start + take
            chunks.append(np.asarray(h5_file["data"][:, real_start:real_end], dtype=np.float32))

            remaining -= take
            virtual_position += take
            interval_idx += 1

        if not chunks:
            first_source = stream["intervals"][0].source_idx
            n_channels = int(self._get_h5(first_source).attrs["n_channels"])
            return np.empty((n_channels, 0), dtype=np.float32)
        return np.concatenate(chunks, axis=1)

    def _read_virtual_segment(self, stream: Dict[str, Any], start: int, length: int) -> np.ndarray:
        total_samples = int(stream["total_samples"])
        if total_samples <= 0:
            raise ValueError(f"Empty stream: {stream['recording_name']}")

        if start + length <= total_samples:
            data = self._read_once(stream, start, length)
            if data.shape[1] != length:
                raise RuntimeError(
                    f"Read {data.shape[1]} samples instead of {length} from "
                    f"{stream['recording_name']}"
                )
            return data

        available = max(0, total_samples - start)
        first = self._read_once(stream, start, available)
        missing = length - first.shape[1]

        if missing <= 0:
            return first[:, :length]
        if self.short_stream_policy == "zero_pad":
            return np.pad(first, ((0, 0), (0, missing)), mode="constant")
        if self.short_stream_policy == "error":
            raise RuntimeError(
                f"Segment exceeds stream {stream['recording_name']} and short_stream_policy=error"
            )

        chunks = [first]
        while missing > 0:
            take = min(missing, total_samples)
            chunks.append(self._read_once(stream, 0, take))
            missing -= take
        return np.concatenate(chunks, axis=1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stream_idx, segment_idx = self.segment_index[idx]
        stream = self.recordings[stream_idx]
        start = self.segment_starts[stream_idx][segment_idx]
        segment_samples = int(round(self.segment_length * self.target_sfreq))

        eeg_data = self._read_virtual_segment(stream, start, segment_samples)
        first_source_idx = stream["intervals"][0].source_idx
        first_h5 = self._get_h5(first_source_idx)
        sensor_xyzdir = np.asarray(first_h5["sensor_xyzdir"][:], dtype=np.float32)
        sensor_types = np.asarray(first_h5["sensor_types"][:], dtype=np.int16)

        eeg_data = preprocess_segment_with_subsegments(
            meg_data=eeg_data,
            sensor_types=sensor_types,
            sfreq=self.target_sfreq,
            subsegment_duration=self.subsegment_duration,
            baseline_duration=self.baseline_duration,
            clip_range=self.clip_range,
        )

        original_channels = eeg_data.shape[0]
        if self.max_channel_dim is not None:
            if original_channels > self.max_channel_dim:
                raise ValueError(
                    f"Recording has {original_channels} channels but max_channel_dim="
                    f"{self.max_channel_dim}"
                )
            pad_channels = self.max_channel_dim - original_channels
            if pad_channels:
                eeg_data = np.pad(eeg_data, ((0, pad_channels), (0, 0)), mode="constant")
                sensor_xyzdir = np.pad(
                    sensor_xyzdir, ((0, pad_channels), (0, 0)), mode="constant"
                )
                sensor_types = np.pad(sensor_types, (0, pad_channels), mode="constant")
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_channels] = 1.0
        else:
            sensor_mask = np.ones(original_channels, dtype=np.float32)

        return {
            "meg": torch.from_numpy(np.asarray(eeg_data, dtype=np.float32)),
            "subject": stream["subject"],
            "session": stream["session"],
            "task": stream["task"],
            "run": stream["run"],
            "sensor_xyzdir": torch.from_numpy(sensor_xyzdir.astype(np.float32, copy=False)),
            "sensor_types": torch.from_numpy(sensor_types.astype(np.int32, copy=False)),
            "sensor_mask": torch.from_numpy(sensor_mask),
            "start_time": float(start / self.target_sfreq),
            "end_time": float((start + segment_samples) / self.target_sfreq),
            "recording_idx": int(stream_idx),
            "segment_idx": int(segment_idx),
            "dataset_name": self.dataset_name,
            "modality": "eeg",
            "content_mode": stream["content_mode"],
            "source_recording_indices": tuple(stream["source_indices"]),
        }

    def get_segment_metadata(self, idx: int) -> Dict[str, Any]:
        stream_idx, segment_idx = self.segment_index[idx]
        stream = self.recordings[stream_idx]
        start = self.segment_starts[stream_idx][segment_idx]
        return {
            "dataset_name": self.dataset_name,
            "subject": stream["subject"],
            "session": stream["session"],
            "task": stream["task"],
            "run": stream["run"],
            "content_mode": stream["content_mode"],
            "start_time": start / self.target_sfreq,
            "end_time": (start + int(round(self.segment_length * self.target_sfreq)))
            / self.target_sfreq,
        }

    def get_split_group(self, idx: int, group_kind: str = "auto") -> str:
        metadata = self.get_segment_metadata(idx)
        if group_kind == "subject":
            return f"{self.dataset_name}:{metadata['subject']}"
        if group_kind == "session":
            return f"{self.dataset_name}:{metadata['subject']}:{metadata['session']}"
        return (
            f"{self.dataset_name}:{metadata['subject']}:{metadata['session']}:"
            f"{metadata['task']}:{metadata['run']}"
        )

    def close(self) -> None:
        for handle in self._file_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._file_handles = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_file_handles"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "IntervalRef",
    "OpenNeuroEEGContinuousDataset",
    "merge_intervals",
    "segment_starts_cover_all",
]
