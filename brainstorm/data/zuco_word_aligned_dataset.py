"""PyTorch Dataset for word-aligned segments from ZuCo 2.0 NR EEG data."""

from __future__ import annotations

import re
import warnings
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.io as sio
import scipy.signal
import torch
from torch.utils.data import Dataset

from .preprocessing import _process_single_chunk
from .utils import norm_sensor_positions


class ZuCoWordAlignedDataset(Dataset):
    """
    Word-aligned dataset for ZuCo 2.0 Natural Reading EEG recordings.

    The ZuCo NR release is not BIDS-like: continuous EEG, corrected eye-tracking,
    text bounding boxes and task material live in separate MATLAB files. This
    adapter aligns first-fixation timestamps to continuous EEG using shared TR
    markers, then exposes the same sample dictionary as the MEG word-aligned
    datasets used by the CrissCross evaluation.

        Each segment contains words_per_segment consecutive words, where each word
    has a subsegment_duration-second EEG window aligned to the first fixation on
    that word. By default, this gives 50 word-aligned windows of 3s each,
    concatenated into a 150s segment. Each 3s subsegment is independently
    resampled and preprocessed with baseline correction, robust scaling, and
    clipping.

    ZuCo stores EEG, eye-tracking events, word bounding boxes, and text
    materials in separate files. This dataset discovers matching Natural Reading
    recordings, extracts the first fixation for each word using the corrected
    eye-tracking data and word bounding boxes, maps eye-tracking timestamps to
    EEG sample indices through shared TR markers, and returns contiguous groups
    of word-aligned EEG windows.

    Although the returned signal key is named "meg", the tensor contains EEG
    data. This is intentional: it keeps the output dictionary compatible with
    the existing MEG word-aligned datasets and CrissCross evaluation code.

    Parameters
    ----------
    data_root : str
        Root directory of the ZuCo 2.0 dataset. The loader expects either a
        directory containing "task1 - NR" directly, or a parent directory
        containing "data/zuco2/task1 - NR".
    segment_length : float
        Total segment length in seconds. For consistency, this should equal
        words_per_segment x subsegment_duration. Default: 150.0.
    subsegment_duration : float
        Duration of each word-aligned EEG window in seconds. Default: 3.0.
    words_per_segment : int
        Number of consecutive word windows concatenated into one sample.
        Default: 50.
    window_onset_offset : float
        Start time of each EEG window relative to the word first-fixation onset,
        in seconds. Default: -0.5, meaning that each window starts 0.5s before
        the fixation onset.
    cache_dir : str, optional
        Directory reserved for cache files. Created if it does not exist.
        Default: "./data/cache".
    subjects : List[str], optional
        List of subjects to include. Subject names are normalized internally
        by uppercasing and removing a possible "sub-" prefix. If None, all
        available subjects are used.
    sessions : List[str], optional
        List of Natural Reading sessions to include, e.g. ["NR1", "NR2"] or
        ["1", "2"]. If None, all available NR sessions are used.
    tasks : List[str], optional
        List of tasks to include. This loader only supports the Natural Reading
        task ("NR"). If None, all discovered NR recordings are used.
    l_freq : float
        Low frequency cutoff stored for compatibility with the MEG dataset
        interface. Default: 0.1 Hz.
    h_freq : float
        High frequency cutoff stored for compatibility with the MEG dataset
        interface. Default: 40.0 Hz.
    target_sfreq : float
        Target sampling frequency after resampling each word window.
        Default: 50.0 Hz.
    channel_filter : callable, optional
        Optional channel filter kept for API compatibility. Default: None.
    max_channel_dim : int, optional
        Maximum channel dimension for padding. If specified, EEG data, sensor
        positions, sensor types, and sensor masks are padded to this number of
        channels. If None, no channel padding is applied.
    baseline_duration : float
        Duration of the baseline window used during preprocessing, in seconds.
        Default: 0.5.
    clip_range : tuple
        Minimum and maximum values used for clipping after scaling.
        Default: (-5, 5).
    eeg_sensor_type : str
        Sensor type label assigned to EEG channels for compatibility with models
        that expect MEG-like sensor type IDs. Supported aliases are "grad",
        "gradiometer", "mag", "meg", "magnetometer", and "eeg".
        Default: "grad".

    Returns (from __getitem__)
    -------
    Dictionary containing:
        - meg: torch.Tensor of shape (n_channels, n_timepoints), containing EEG
          data despite the MEG-compatible key name
        - words: List[str] of length words_per_segment
        - subsegment_boundaries: List[Dict] with 'start_sample' and 'end_sample'
          keys for each word window in the concatenated segment
        - sensor_xyzdir: torch.Tensor of shape (n_channels, 6), containing sensor
          positions and normalized directions
        - sensor_types: torch.Tensor of shape (n_channels,)
        - sensor_mask: torch.Tensor of shape (n_channels,), with 1 for real
          channels and 0 for padded channels
        - subject: str
        - session: str
        - task: str, always "NR"
        - recording_idx: int
        - segment_idx: int
        - start_time: float, start time of the first word window in seconds
        - end_time: float, end time of the last word window in seconds

    Example
    -------
    >>> dataset = ZuCoWordAlignedDataset(
    ...     data_root="/path/to/zuco2",
    ...     segment_length=150.0,
    ...     subsegment_duration=3.0,
    ...     words_per_segment=50,
    ...     window_onset_offset=-0.5,
    ...     sessions=["NR1"],
    ...     target_sfreq=50.0,
    ... )
    >>> print(f"Dataset: {len(dataset)} segments")
    >>> sample = dataset[0]
    >>> print(f"EEG shape: {sample['meg'].shape}")
    >>> print(f"Words: {sample['words']}")
    >>> print(f"Number of subsegments: {len(sample['subsegment_boundaries'])}")
    """

    _SESSION_RE = re.compile(r"NR(\d+)", re.IGNORECASE)
    _EEG_RE = re.compile(r"^[a-z]ip_(?P<subject>[^_]+)_NR(?P<nr>\d+)_EEG\.mat$")
    _TR_RE = re.compile(r"MSG\s+(?P<time>\d+)\s+TR(?P<code>\d+)")
    _EFIX_RE = re.compile(
        r"EFIX\s+\S+\s+(?P<start>\d+)\s+(?P<end>\d+)\s+\S+\s+"
        r"(?P<x>[-+]?\d+(?:\.\d+)?)\s+(?P<y>[-+]?\d+(?:\.\d+)?)"
    )
    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

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
        channel_filter=None,
        max_channel_dim: Optional[int] = None,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5),
        eeg_sensor_type: str = "grad",
    ):
        self.data_root = self._resolve_data_root(Path(data_root))
        self.preprocessed_root = self.data_root / "task1 - NR" / "Preprocessed"
        self.task_materials_root = self.data_root / "task_materials"

        self.segment_length = segment_length
        self.subsegment_duration = subsegment_duration
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_duration = baseline_duration
        self.clip_range = clip_range
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.eeg_sensor_type = eeg_sensor_type
        self.eeg_sensor_type_id = self._resolve_eeg_sensor_type(eeg_sensor_type)

        self.subjects = [self._normalize_subject(s) for s in subjects] if subjects is not None else None
        self.sessions = [self._normalize_session(s) for s in sessions] if sessions is not None else None
        self.tasks = [str(task).upper() for task in tasks] if tasks is not None else None

        self.recordings = self._discover_recordings()
        if len(self.recordings) == 0:
            raise ValueError(
                f"No ZuCo recordings found in {self.preprocessed_root}. "
                f"Subjects: {subjects}, Sessions: {sessions}, Tasks: {tasks}"
            )

        self.file_handles: List[h5py.File] = []
        self.word_groups: List[List[List[Dict[str, Any]]]] = []
        self._open_file_handles()
        self._parse_all_recordings()
        self.segment_index = self._build_segment_index()

    @staticmethod
    def _resolve_data_root(path: Path) -> Path:
        if (path / "task1 - NR").exists():
            return path
        nested = path / "data" / "zuco2"
        if (nested / "task1 - NR").exists():
            return nested
        raise FileNotFoundError(
            f"Could not find ZuCo 'task1 - NR' under {path} or {nested}"
        )

    @classmethod
    def _normalize_session(cls, session: str) -> str:
        match = cls._SESSION_RE.search(str(session))
        if match:
            return f"NR{int(match.group(1))}"
        text = str(session).strip().upper()
        if text.isdigit():
            return f"NR{int(text)}"
        return text

    @staticmethod
    def _normalize_subject(subject: str) -> str:
        text = str(subject).strip()
        return text.upper().replace("SUB-", "")

    @staticmethod
    def _resolve_eeg_sensor_type(sensor_type: str) -> int:
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
                f"Unknown eeg_sensor_type={sensor_type!r}. "
                f"Expected one of: {sorted(aliases.keys())}"
            )
        return aliases[key]

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        recordings: List[Dict[str, Any]] = []

        if self.tasks is not None and "NR" not in self.tasks:
            return recordings

        subject_dirs = sorted(p for p in self.preprocessed_root.iterdir() if p.is_dir())
        for subject_dir in subject_dirs:
            subject = self._normalize_subject(subject_dir.name)
            if self.subjects is not None and subject not in self.subjects:
                continue

            for eeg_path in sorted(subject_dir.glob("*_NR*_EEG.mat")):
                match = self._EEG_RE.match(eeg_path.name)
                if match is None:
                    continue

                session = f"NR{int(match.group('nr'))}"
                if self.sessions is not None and session not in self.sessions:
                    continue

                et_path = subject_dir / f"{subject}_NR{session[2:]}_corrected_ET.mat"
                wordbounds_path = self.preprocessed_root / f"wordbounds_NR{session[2:]}.mat"
                task_materials_path = self.task_materials_root / f"nr_{session[2:]}.csv"

                missing = [
                    path for path in (et_path, wordbounds_path, task_materials_path)
                    if not path.exists()
                ]
                if missing:
                    warnings.warn(
                        f"Skipping {subject} {session}; missing files: "
                        f"{', '.join(str(p) for p in missing)}"
                    )
                    continue

                if not h5py.is_hdf5(eeg_path):
                    warnings.warn(
                        f"Skipping {subject} {session}; EEG file is not a readable "
                        f"MATLAB v7.3/HDF5 file: {eeg_path}"
                    )
                    continue

                recordings.append({
                    "subject": subject,
                    "session": session,
                    "task": "NR",
                    "raw_path": eeg_path,
                    "et_path": et_path,
                    "wordbounds_path": wordbounds_path,
                    "task_materials_path": task_materials_path,
                })

        return recordings

    def _open_file_handles(self) -> None:
        self.file_handles = []
        for rec in self.recordings:
            h5_file = h5py.File(rec["raw_path"], "r")
            rec["sample_freq"] = float(h5_file["EEG/srate"][0, 0])
            rec["n_samples"] = int(h5_file["EEG/pnts"][0, 0])
            rec["channel_names"] = self._read_channel_names(h5_file)
            rec["sensor_xyzdir"] = self._read_sensor_xyzdir(h5_file)
            rec["sensor_types"] = np.full(
                len(rec["channel_names"]),
                self.eeg_sensor_type_id,
                dtype=np.int64,
            )
            self.file_handles.append(h5_file)

    def _read_channel_names(self, h5_file: h5py.File) -> List[str]:
        labels = h5_file["EEG/chanlocs/labels"]
        names = []
        for idx in range(labels.shape[0]):
            arr = h5_file[labels[idx, 0]][()]
            names.append("".join(chr(int(c)) for c in arr.flat))
        return names

    def _read_ref_scalar_array(self, h5_file: h5py.File, path: str) -> np.ndarray:
        refs = h5_file[path]
        values = []
        for idx in range(refs.shape[0]):
            arr = np.asarray(h5_file[refs[idx, 0]][()]).squeeze()
            values.append(float(arr) if arr.size != 0 else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _read_sensor_xyzdir(self, h5_file: h5py.File) -> np.ndarray:
        x = self._read_ref_scalar_array(h5_file, "EEG/chanlocs/X")
        y = self._read_ref_scalar_array(h5_file, "EEG/chanlocs/Y")
        z = self._read_ref_scalar_array(h5_file, "EEG/chanlocs/Z")
        xyz = np.stack([x, y, z], axis=1)
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        directions = np.divide(
            xyz,
            norms,
            out=np.zeros_like(xyz, dtype=np.float32),
            where=norms > 0,
        )
        return np.concatenate([xyz, directions], axis=1).astype(np.float32)

    def _parse_all_recordings(self) -> None:
        self.word_groups = []
        for rec_idx, rec in enumerate(self.recordings):
            word_events = self._build_word_events(rec_idx, rec)
            groups = self._build_word_groups(word_events, rec)
            self.word_groups.append(groups)
            print(
                f"Recording {rec_idx} ({rec['subject']} {rec['session']}): "
                f"Found {len(groups)} word-aligned segments"
            )

    def _load_page_words(self, task_materials_path: Path, page_lengths: List[int]) -> List[List[str]]:
        tokens: List[str] = []
        for line in task_materials_path.read_text(errors="replace").splitlines():
            if line.endswith(";CONTROL"):
                line = line[:-len(";CONTROL")]
            parts = line.split(";", 2)
            if len(parts) != 3:
                continue
            tokens.extend(t.lower() for t in self._TOKEN_RE.findall(parts[2]))

        page_words: List[List[str]] = []
        cursor = 0
        for page_len in page_lengths:
            page = tokens[cursor:cursor + page_len]
            if len(page) < page_len:
                warnings.warn(
                    f"Task materials ended early for {task_materials_path}; "
                    f"expected {page_len} words, got {len(page)}."
                )
            page_words.append(page)
            cursor += page_len
        return page_words

    def _load_wordbounds(self, wordbounds_path: Path) -> List[np.ndarray]:
        mat = sio.loadmat(wordbounds_path, squeeze_me=True, struct_as_record=False)
        wordbounds = mat["wordbounds"]
        return [np.asarray(bounds, dtype=np.float32).reshape(-1, 4) for bounds in wordbounds]

    def _parse_et_markers(self, et_path: Path) -> Tuple[List[Tuple[int, int]], List[Dict[str, float]]]:
        mat = sio.loadmat(et_path, squeeze_me=True, struct_as_record=False)
        markers: List[Tuple[int, int]] = []
        fixations: List[Dict[str, float]] = []

        for raw_message in mat["messages"]:
            message = str(raw_message).strip()

            marker = self._TR_RE.search(message)
            if marker:
                markers.append((int(marker.group("time")), int(marker.group("code"))))
                continue

            fixation = self._EFIX_RE.search(message)
            if fixation:
                fixations.append({
                    "start": float(fixation.group("start")),
                    "end": float(fixation.group("end")),
                    "x": float(fixation.group("x")),
                    "y": float(fixation.group("y")),
                })

        return markers, fixations

    def _read_eeg_markers(self, h5_file: h5py.File) -> List[Tuple[float, int]]:
        event = h5_file["EEG/event"]
        latencies = event["latency"]
        types = event["type"]
        markers: List[Tuple[float, int]] = []

        for idx in range(latencies.shape[0]):
            latency = float(np.asarray(h5_file[latencies[idx, 0]][()]).squeeze())
            type_arr = h5_file[types[idx, 0]][()]
            type_text = "".join(chr(int(c)) for c in type_arr.flat).strip()
            if type_text.isdigit():
                markers.append((latency, int(type_text)))

        return markers

    def _fit_et_to_eeg_samples(
        self,
        et_markers: List[Tuple[int, int]],
        eeg_markers: List[Tuple[float, int]],
    ) -> Tuple[float, float]:
        n = min(len(et_markers), len(eeg_markers))
        pairs = [
            (et_markers[i][0], eeg_markers[i][0])
            for i in range(n)
            if et_markers[i][1] == eeg_markers[i][1]
        ]
        if len(pairs) < 2:
            raise ValueError("Could not align ZuCo ET and EEG markers")

        et_times = np.asarray([p[0] for p in pairs], dtype=np.float64)
        eeg_samples = np.asarray([p[1] for p in pairs], dtype=np.float64)
        slope, intercept = np.polyfit(et_times, eeg_samples, 1)
        return float(slope), float(intercept)

    def _text_trials(self, et_markers: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        trials: List[Tuple[int, int]] = []
        pending_start: Optional[int] = None

        for timestamp, code in et_markers:
            if code == 10:
                pending_start = timestamp
            elif code == 11 and pending_start is not None:
                trials.append((pending_start, timestamp))
                pending_start = None

        return trials

    def _first_fixations_by_word(
        self,
        bounds: np.ndarray,
        fixations: List[Dict[str, float]],
        margin: float = 3.0,
    ) -> Dict[int, Dict[str, float]]:
        first_by_word: Dict[int, Dict[str, float]] = {}

        for fixation in fixations:
            x = fixation["x"]
            y = fixation["y"]
            hits = np.where(
                (x >= bounds[:, 0] - margin)
                & (x <= bounds[:, 2] + margin)
                & (y >= bounds[:, 1] - margin)
                & (y <= bounds[:, 3] + margin)
            )[0]
            if len(hits) == 0:
                continue

            word_idx = int(hits[0])
            if word_idx not in first_by_word:
                first_by_word[word_idx] = fixation

        return first_by_word

    def _build_word_events(self, rec_idx: int, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        h5_file = self.file_handles[rec_idx]
        wordbounds = self._load_wordbounds(rec["wordbounds_path"])
        page_words = self._load_page_words(rec["task_materials_path"], [len(b) for b in wordbounds])
        et_markers, fixations = self._parse_et_markers(rec["et_path"])
        eeg_markers = self._read_eeg_markers(h5_file)
        slope, intercept = self._fit_et_to_eeg_samples(et_markers, eeg_markers)
        text_trials = self._text_trials(et_markers)

        n_pages = min(len(wordbounds), len(page_words), len(text_trials))
        if n_pages == 0:
            warnings.warn(
                f"{rec['subject']} {rec['session']}: no aligned ZuCo text trials found."
            )

        word_events: List[Dict[str, Any]] = []
        orig_sfreq = rec["sample_freq"]
        n_samples = rec["n_samples"]

        for page_idx in range(n_pages):
            trial_start, trial_end = text_trials[page_idx]
            page_fixations = [
                fixation for fixation in fixations
                if trial_start <= fixation["start"] <= trial_end
            ]
            first_by_word = self._first_fixations_by_word(wordbounds[page_idx], page_fixations)

            for word_idx, word in enumerate(page_words[page_idx]):
                fixation = first_by_word.get(word_idx)
                if fixation is None:
                    continue

                onset_sample = int(round(slope * fixation["start"] + intercept))
                start_sample = int(round(onset_sample + self.window_onset_offset * orig_sfreq))
                end_sample = start_sample + int(round(self.subsegment_duration * orig_sfreq))

                if start_sample < 0 or end_sample > n_samples:
                    continue

                word_events.append({
                    "word": word,
                    "onset": onset_sample / orig_sfreq,
                    "window_start": start_sample / orig_sfreq,
                    "window_end": end_sample / orig_sfreq,
                    "start_sample_orig": start_sample,
                    "end_sample_orig": end_sample,
                    "page_idx": page_idx,
                    "word_idx": word_idx,
                })

        word_events.sort(key=lambda item: item["start_sample_orig"])
        return word_events

    def _build_word_groups(
        self,
        word_events: List[Dict[str, Any]],
        rec: Dict[str, Any],
    ) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []

        for event in word_events:
            item = dict(event)
            item["subsegment_idx"] = len(current_group)
            current_group.append(item)

            if len(current_group) == self.words_per_segment:
                groups.append(current_group.copy())
                current_group = []

        return groups

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        segment_index = []
        for rec_idx, groups in enumerate(self.word_groups):
            for group_idx in range(len(groups)):
                segment_index.append((rec_idx, group_idx))
        return segment_index

    def __len__(self) -> int:
        return len(self.segment_index)

    def _resample_subsegment(self, chunk: np.ndarray, orig_sfreq: float) -> np.ndarray:
        expected_samples = int(round(self.subsegment_duration * self.target_sfreq))
        if abs(orig_sfreq - self.target_sfreq) <= 0.1:
            resampled = chunk
        else:
            orig_int = int(round(orig_sfreq))
            target_int = int(round(self.target_sfreq))
            divisor = gcd(orig_int, target_int)
            up = target_int // divisor
            down = orig_int // divisor
            resampled = scipy.signal.resample_poly(chunk, up=up, down=down, axis=1)

        if resampled.shape[1] > expected_samples:
            resampled = resampled[:, :expected_samples]
        elif resampled.shape[1] < expected_samples:
            pad = expected_samples - resampled.shape[1]
            resampled = np.pad(resampled, ((0, 0), (0, pad)))

        return resampled.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, group_idx = self.segment_index[idx]
        h5_file = self.file_handles[rec_idx]
        rec = self.recordings[rec_idx]
        word_group = self.word_groups[rec_idx][group_idx]

        data = h5_file["EEG/data"]
        sensor_types = rec["sensor_types"]
        subsegments = []

        for word_info in word_group:
            start = word_info["start_sample_orig"]
            end = word_info["end_sample_orig"]
            eeg_subsegment = np.asarray(data[start:end, :], dtype=np.float32).T
            eeg_subsegment = np.nan_to_num(eeg_subsegment, copy=False)
            eeg_subsegment = self._resample_subsegment(eeg_subsegment, rec["sample_freq"])
            processed = _process_single_chunk(
                eeg_subsegment,
                sensor_types,
                self.target_sfreq,
                self.baseline_duration,
                self.clip_range,
            )
            subsegments.append(processed)

        eeg_data = np.concatenate(subsegments, axis=1)
        sensor_xyzdir = norm_sensor_positions(rec["sensor_xyzdir"].copy())
        sensor_types = sensor_types.copy()

        if self.max_channel_dim is not None:
            original_n_channels = eeg_data.shape[0]
            if original_n_channels > self.max_channel_dim:
                raise ValueError(
                    f"ZuCo recording has {original_n_channels} channels, "
                    f"but max_channel_dim={self.max_channel_dim}"
                )
            eeg_data = np.pad(eeg_data, ((0, self.max_channel_dim - original_n_channels), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - sensor_types.shape[0]))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_n_channels] = 1.0
        else:
            sensor_mask = np.ones(eeg_data.shape[0], dtype=np.float32)

        words = [w["word"] for w in word_group]
        subsegment_boundaries = []
        cumulative_samples = 0
        for subsegment in subsegments:
            subsegment_boundaries.append({
                "start_sample": cumulative_samples,
                "end_sample": cumulative_samples + subsegment.shape[1],
            })
            cumulative_samples += subsegment.shape[1]

        return {
            "meg": torch.from_numpy(eeg_data).float(),
            "subject": rec["subject"],
            "session": rec["session"],
            "task": rec["task"],
            "sensor_xyzdir": torch.from_numpy(sensor_xyzdir).float(),
            "sensor_types": torch.from_numpy(sensor_types).int(),
            "sensor_mask": torch.from_numpy(sensor_mask).float(),
            "words": words,
            "subsegment_boundaries": subsegment_boundaries,
            "recording_idx": rec_idx,
            "segment_idx": group_idx,
            "start_time": float(word_group[0]["window_start"]),
            "end_time": float(word_group[-1]["window_end"]),
        }

    def __del__(self):
        self.close()

    def close(self):
        for h5_file in getattr(self, "file_handles", []):
            try:
                h5_file.close()
            except Exception:
                pass
        self.file_handles = []
