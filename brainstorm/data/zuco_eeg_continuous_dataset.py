"""Continuous ZuCo 2.0 Natural Reading EEG dataset for pre-training."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import mne
import numpy as np

from .eeg_word_aligned_dataset import EEGChannelCount
from .openneuro_eeg_continuous_dataset import (
    EEG_SENSOR_TYPE_ID,
    OpenNeuroEEGContinuousDataset,
)

ZUCO_TASK = "NR"
_ZUCO_RE = re.compile(
    r"^[a-z]+_(?P<subject>[^_]+)_NR(?P<session>\d+)_EEG\.mat$",
    re.IGNORECASE,
)


def _resolve_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if (root / "task1 - NR" / "Preprocessed").is_dir():
        return root
    nested = root / "data" / "zuco2"
    return nested if (nested / "task1 - NR" / "Preprocessed").is_dir() else root


def _recordings(root: Path):
    directory = root / "task1 - NR" / "Preprocessed"
    if not directory.is_dir():
        return
    for subject_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for path in sorted(subject_dir.glob("*_NR*_EEG.mat")):
            match = _ZUCO_RE.match(path.name)
            if match is not None:
                yield path, match


def _scalar(h5_file: h5py.File, path: str) -> float:
    value = np.asarray(h5_file[path][()]).squeeze()
    if value.size == 0:
        raise ValueError(f"Empty ZuCo field {path} in {h5_file.filename}")
    return float(value.flat[0])


def _dereference(h5_file: h5py.File, value: Any) -> np.ndarray:
    return np.asarray(h5_file[value][()]) if isinstance(value, h5py.Reference) else np.asarray(value)


def _reference_values(h5_file: h5py.File, path: str) -> np.ndarray:
    values = []
    for reference in np.asarray(h5_file[path][()]).reshape(-1):
        array = _dereference(h5_file, reference).squeeze()
        values.append(float(array.flat[0]) if array.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _channel_names(h5_file: h5py.File) -> List[str]:
    names = []
    for reference in np.asarray(h5_file["EEG/chanlocs/labels"][()]).reshape(-1):
        array = _dereference(h5_file, reference).reshape(-1)
        names.append("".join(chr(int(value)) for value in array if int(value) > 0).strip())
    return names


def _sensor_xyzdir(h5_file: h5py.File) -> np.ndarray:
    xyz = np.stack(
        [_reference_values(h5_file, f"EEG/chanlocs/{axis}") for axis in "XYZ"],
        axis=1,
    )
    xyz = np.nan_to_num(xyz, copy=False)
    directions = np.divide(
        xyz,
        np.linalg.norm(xyz, axis=1, keepdims=True),
        out=np.zeros_like(xyz),
        where=np.linalg.norm(xyz, axis=1, keepdims=True) > 0,
    )
    centered = xyz - np.mean(xyz, axis=0, keepdims=True)
    scale = float(np.sqrt(3.0 * np.mean(np.sum(centered**2, axis=1))))
    normalized = centered / scale if np.isfinite(scale) and scale > 0 else np.zeros_like(xyz)
    return np.concatenate([normalized, directions], axis=1).astype(np.float32)


def scan_zuco_eeg_channel_counts(
    data_root: str | Path,
    tasks: Optional[Sequence[str]] = None,
) -> List[EEGChannelCount]:
    """Return channel counts from readable MATLAB v7.3 recordings."""

    if tasks is not None and ZUCO_TASK.lower() not in {str(task).lower() for task in tasks}:
        return []
    counts = []
    for path, _ in _recordings(_resolve_root(data_root)) or ():
        if not h5py.is_hdf5(path):
            continue
        try:
            with h5py.File(path, "r") as h5_file:
                count = int(np.asarray(h5_file["EEG/chanlocs/labels"]).size)
            counts.append(EEGChannelCount(path, count, "zuco hdf5 metadata"))
        except Exception as exc:
            warnings.warn(f"Could not inspect {path}: {exc}", RuntimeWarning)
    return counts


class ZuCoEEGContinuousDataset(OpenNeuroEEGContinuousDataset):
    """Return fixed continuous windows from ZuCo 2.0 Natural Reading EEG.

    ZuCo stores samples in MATLAB/EEGLAB files with shape ``time x channels``.
    The loader reads each v7.3/HDF5 recording, keeps its continuous timeline,
    filters and resamples it, then writes the standard continuous EEG cache used
    by :class:`OpenNeuroEEGContinuousDataset`. The two legacy non-HDF5 files in
    the local release are skipped with a warning.
    """

    def __init__(
        self,
        data_root: str | Path,
        *args: Any,
        dataset_name: str = "zuco",
        tasks: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_root=str(_resolve_root(data_root)),
            dataset_name=dataset_name,
            *args,
            tasks=[ZUCO_TASK] if tasks is None else tasks,
            **kwargs,
        )

    def _discover_source_recordings(self) -> List[Dict[str, Any]]:
        found = []
        if self.tasks is not None and ZUCO_TASK.lower() not in self.tasks:
            return found
        for path, match in _recordings(self.data_root) or ():
            subject = match.group("subject")
            session = f"NR{int(match.group('session'))}"
            if self.subjects is not None and subject.lower() not in self.subjects:
                continue
            if self.sessions is not None and session.lower() not in self.sessions:
                continue
            if not h5py.is_hdf5(path):
                warnings.warn(f"Skipping non-HDF5 ZuCo file: {path}", RuntimeWarning)
                continue
            recording = {
                "raw_path": path,
                "events_path": None,
                "subject": f"sub-{subject}",
                "session": f"ses-{session}",
                "task": ZUCO_TASK,
                "run": "",
                "entities": {"sub": subject, "ses": session, "task": ZUCO_TASK},
            }
            recording["cache_path"] = self._cache_path(recording)
            found.append(recording)
        return found

    def _preprocess_recording(self, recording: Dict[str, Any]) -> None:
        source = Path(recording["raw_path"])
        with h5py.File(source, "r") as h5_file:
            required = [
                "EEG/data", "EEG/srate", "EEG/pnts", "EEG/chanlocs/labels",
                "EEG/chanlocs/X", "EEG/chanlocs/Y", "EEG/chanlocs/Z",
            ]
            missing = [path for path in required if path not in h5_file]
            if missing:
                raise ValueError(f"{source} is missing ZuCo fields: {missing}")
            sfreq = _scalar(h5_file, "EEG/srate")
            n_samples = int(round(_scalar(h5_file, "EEG/pnts")))
            names = _channel_names(h5_file)
            positions = _sensor_xyzdir(h5_file)
            stored = h5_file["EEG/data"]
            if stored.shape == (n_samples, len(names)):
                data = np.asarray(stored, dtype=np.float32).T
            elif stored.shape == (len(names), n_samples):
                data = np.asarray(stored, dtype=np.float32)
            else:
                raise ValueError(
                    f"Unexpected EEG/data shape {stored.shape} in {source}; "
                    f"expected {n_samples} samples and {len(names)} channels"
                )

        if self.channel_filter is not None:
            keep = [index for index, name in enumerate(names) if self.channel_filter(name)]
            if not keep:
                raise ValueError(f"Channel filter removed every channel in {source}")
            names = [names[index] for index in keep]
            data = data[keep]
            positions = positions[keep]

        raw = mne.io.RawArray(
            np.nan_to_num(data, copy=False),
            mne.create_info(names, sfreq=sfreq, ch_types="eeg"),
            verbose=False,
        )
        try:
            nyquist = sfreq / 2.0
            if self.h_freq >= nyquist:
                raise ValueError(f"h_freq={self.h_freq} must be below {nyquist:g} Hz")
            raw.filter(self.l_freq, self.h_freq, picks="eeg", n_jobs=1, verbose=False)
            if not math.isclose(sfreq, self.target_sfreq, abs_tol=1e-6):
                raw.resample(self.target_sfreq, n_jobs=1, verbose=False)
            data = raw.get_data().astype(np.float32, copy=False)
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()

        cache = Path(recording["cache_path"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, "w") as h5_file:
            h5_file.create_dataset("data", data=data, compression="lzf")
            h5_file.create_dataset("sensor_xyzdir", data=positions)
            h5_file.create_dataset(
                "sensor_types",
                data=np.full(data.shape[0], EEG_SENSOR_TYPE_ID, dtype=np.int16),
            )
            h5_file.create_dataset(
                "channel_names",
                data=np.asarray(names, dtype=h5py.string_dtype("utf-8")),
            )
            h5_file.attrs["n_samples"] = data.shape[1]
            h5_file.attrs["n_channels"] = data.shape[0]
            h5_file.attrs["sample_freq"] = self.target_sfreq
            h5_file.attrs["subject"] = recording["subject"]
            h5_file.attrs["session"] = recording["session"]
            h5_file.attrs["task"] = recording["task"]
            h5_file.attrs["run"] = recording["run"]
            h5_file.attrs["raw_path"] = str(source)
            h5_file.attrs["channels_path"] = "EEG/chanlocs"
            h5_file.attrs["l_freq"] = self.l_freq
            h5_file.attrs["h_freq"] = self.h_freq
            h5_file.attrs["original_sample_freq"] = sfreq
        temporary.replace(cache)
        print(f"{source.name}: {data.shape[0]} EEG channels, {sfreq:g} -> {self.target_sfreq:g} Hz")


__all__ = ["ZUCO_TASK", "ZuCoEEGContinuousDataset", "scan_zuco_eeg_channel_counts"]
