"""Continuous SparrKULee EEG dataset for MEG-XL-style pre-training.

SparrKULee is distributed as a BIDS dataset with ``task-listeningActive``
recordings stored as BioSemi BDF files. Older sessions can be stored as
``.bdf.gz`` files, so this loader materializes one compressed recording at a
time, preprocesses it through the shared continuous EEG pipeline, and removes
the temporary BDF afterwards.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mne
import pandas as pd

from .eeg_word_aligned_dataset import EEGChannelCount
from .openneuro_eeg_continuous_dataset import OpenNeuroEEGContinuousDataset


SPARRKULEE_TASK = "listeningActive"
_RAW_ENDINGS = ("_eeg.bdf", "_eeg.bdf.gz")


def _entity_dict(path: Path) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    for part in path.name.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        entities[key] = value.split(".")[0]
    return entities


def _is_sparrkulee_raw(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(_RAW_ENDINGS)


def _sidecar_path(raw_path: Path, suffix: str) -> Path:
    marker = "_eeg."
    if marker not in raw_path.name:
        raise ValueError(f"Cannot infer BIDS sidecar from {raw_path}")
    bids_prefix = raw_path.name.split(marker, maxsplit=1)[0]
    return raw_path.with_name(f"{bids_prefix}_{suffix}")


def _logical_bdf_key(path: Path) -> str:
    text = str(path)
    return text[:-3] if text.lower().endswith(".gz") else text


def _discover_raw_paths(root: Path) -> List[Path]:
    """Return unique raw BDF recordings, preferring uncompressed files."""

    candidates: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not _is_sparrkulee_raw(path):
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if relative_parts and relative_parts[0].lower() in {"derivatives", "stimuli"}:
            continue

        key = _logical_bdf_key(path)
        previous = candidates.get(key)
        if previous is None or (
            previous.name.lower().endswith(".bdf.gz")
            and path.name.lower().endswith(".bdf")
        ):
            candidates[key] = path
    return sorted(candidates.values())


def scan_sparrkulee_eeg_channel_counts(
    data_root: str | Path,
    tasks: Optional[Sequence[str]] = None,
) -> List[EEGChannelCount]:
    """Read SparrKULee EEG channel counts without decompressing BDF files."""

    root = Path(data_root)
    if not root.exists():
        return []

    task_filter = {str(task).lower() for task in tasks} if tasks is not None else None
    counts: List[EEGChannelCount] = []
    seen_sidecars = set()

    for raw_path in _discover_raw_paths(root):
        entities = _entity_dict(raw_path)
        task = entities.get("task", "").lower()
        if task_filter is not None and task not in task_filter:
            continue

        channels_path = _sidecar_path(raw_path, "channels.tsv")
        if channels_path in seen_sidecars or not channels_path.exists():
            continue
        seen_sidecars.add(channels_path)

        try:
            channels = pd.read_csv(channels_path, sep="\t")
            if "type" in channels.columns:
                eeg_rows = channels["type"].astype(str).str.upper().eq("EEG")
                n_channels = int(eeg_rows.sum())
            else:
                n_channels = int(len(channels))
        except Exception:
            continue

        if n_channels > 0:
            counts.append(EEGChannelCount(channels_path, n_channels, "channels.tsv"))

    return counts


class SparrKULeeEEGContinuousDataset(OpenNeuroEEGContinuousDataset):
    """Continuous SparrKULee loader using the shared EEG preprocessing path."""

    def __init__(
        self,
        *args: Any,
        tasks: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        if tasks is None:
            tasks = [SPARRKULEE_TASK]
        super().__init__(*args, tasks=tasks, **kwargs)

    def _discover_source_recordings(self) -> List[Dict[str, Any]]:
        recordings: List[Dict[str, Any]] = []

        for raw_path in _discover_raw_paths(self.data_root):
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

            subject = f"sub-{subject_id}" if subject_id else "sub-unknown"
            session = f"ses-{session_id}" if session_id else ""
            events_path = _sidecar_path(raw_path, "events.tsv")
            channels_path = _sidecar_path(raw_path, "channels.tsv")

            recording: Dict[str, Any] = {
                "raw_path": raw_path,
                "channels_path": channels_path,
                "events_path": events_path if events_path.exists() else None,
                "subject": subject,
                "session": session,
                "task": task or SPARRKULEE_TASK.lower(),
                "run": run,
                "entities": entities,
            }
            recording["cache_path"] = self._cache_path(recording)
            recordings.append(recording)

        return recordings

    def _temporary_bdf_path(self, compressed_path: Path) -> Path:
        try:
            stat = compressed_path.stat()
            identity = f"{compressed_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            identity = str(compressed_path.resolve())
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        stem = compressed_path.name[:-7]  # remove .bdf.gz
        materialized_dir = self.cache_dir / "materialized_bdf"
        materialized_dir.mkdir(parents=True, exist_ok=True)
        return materialized_dir / f"{stem}_{digest}.bdf"

    def _preprocess_recording(self, recording: Dict[str, Any]) -> None:
        raw_path = Path(recording["raw_path"])
        if not raw_path.name.lower().endswith(".bdf.gz"):
            super()._preprocess_recording(recording)
            return

        materialized_path = self._temporary_bdf_path(raw_path)
        temporary_path = materialized_path.with_suffix(".bdf.tmp")
        try:
            if temporary_path.exists():
                temporary_path.unlink()
            with gzip.open(raw_path, "rb") as source, open(temporary_path, "wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            temporary_path.replace(materialized_path)

            prepared = dict(recording)
            prepared["raw_path"] = materialized_path
            super()._preprocess_recording(prepared)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
            if materialized_path.exists():
                materialized_path.unlink()

    @staticmethod
    def _apply_eeg_montage(raw: mne.io.BaseRaw, raw_path: Path) -> None:
        montage = mne.channels.make_standard_montage("biosemi64")
        montage_names = set(montage.ch_names)

        # BIDS conversions may retain either the standard 10-20 labels or the
        # original BioSemi acquisition labels A1-A32/B1-B32. MNE's biosemi64
        # montage is ordered in that same acquisition order.
        if set(raw.ch_names) == montage_names:
            raw.set_montage(montage, match_case=True, on_missing="raise")
            print("Applied SparrKULee BioSemi 64 montage")
            return

        acquisition_labels = [
            f"{group}{index}"
            for group in ("A", "B")
            for index in range(1, 33)
        ]
        normalized_to_original = {}
        for name in raw.ch_names:
            stripped = name.strip()
            if len(stripped) < 2 or stripped[0].upper() not in {"A", "B"}:
                continue
            try:
                normalized = f"{stripped[0].upper()}{int(stripped[1:])}"
            except ValueError:
                continue
            normalized_to_original[normalized] = name

        if set(normalized_to_original) == set(acquisition_labels):
            label_to_montage = dict(zip(acquisition_labels, montage.ch_names))
            raw.rename_channels(
                {
                    original: label_to_montage[label]
                    for label, original in normalized_to_original.items()
                }
            )
            raw.set_montage(montage, match_case=True, on_missing="raise")
            print("Applied SparrKULee BioSemi 64 montage")
            return

        OpenNeuroEEGContinuousDataset._apply_eeg_montage(raw, raw_path)


__all__ = [
    "SPARRKULEE_TASK",
    "SparrKULeeEEGContinuousDataset",
    "scan_sparrkulee_eeg_channel_counts",
]
