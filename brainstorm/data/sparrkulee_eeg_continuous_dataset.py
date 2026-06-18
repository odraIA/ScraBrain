"""Continuous SparrKULee EEG dataset for MEG-XL-style pre-training.

SparrKULee is distributed as a BIDS dataset with ``task-listeningActive``
recordings stored as BioSemi BDF files. Older sessions can be stored as
``.bdf.gz`` files, so this loader materializes one compressed recording at a
time, preprocesses it through the shared continuous EEG pipeline, and removes
the temporary BDF afterwards.

Unlike the OpenNeuro datasets used by this project, SparrKULee does not always
provide a run-level ``*_channels.tsv``. This loader follows BIDS inheritance
when a less-specific sidecar exists and otherwise reconstructs channel types
from the known BioSemi-64 acquisition layout.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mne
import pandas as pd

from .eeg_word_aligned_dataset import EEGChannelCount
from .openneuro_eeg_continuous_dataset import OpenNeuroEEGContinuousDataset


SPARRKULEE_TASK = "listeningActive"
SPARRKULEE_EEG_CHANNELS = 64
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


def _is_ancestor(path: Path, possible_ancestor: Path) -> bool:
    try:
        path.relative_to(possible_ancestor)
        return True
    except ValueError:
        return False


def _resolve_bids_sidecar(
    data_root: Path,
    raw_path: Path,
    suffix: str,
) -> Optional[Path]:
    """Resolve a sidecar using the BIDS inheritance principle.

    The exact run-level sidecar is preferred. Otherwise, files in the EEG,
    session, subject and dataset directories are considered when all entities
    encoded in the sidecar also match the raw recording.
    """

    exact = _sidecar_path(raw_path, suffix)
    if exact.exists():
        return exact
    exact_gz = exact.with_suffix(exact.suffix + ".gz")
    if exact_gz.exists():
        return exact_gz

    raw_entities = _entity_dict(raw_path)
    candidates = []
    current = raw_path.parent
    distance = 0

    while _is_ancestor(current, data_root):
        for candidate in list(current.glob(f"*_{suffix}")) + list(
            current.glob(f"*_{suffix}.gz")
        ):
            candidate_entities = _entity_dict(candidate)
            if any(
                raw_entities.get(key, "").lower() != value.lower()
                for key, value in candidate_entities.items()
                if key in {"sub", "ses", "task", "acq", "run"}
            ):
                continue
            specificity = sum(
                key in candidate_entities
                for key in ("sub", "ses", "task", "acq", "run")
            )
            candidates.append((specificity, -distance, len(candidate.name), candidate))

        if current == data_root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
        distance += 1

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[:3])[3]


def _count_eeg_rows(channels_path: Path) -> Optional[int]:
    try:
        channels = pd.read_csv(channels_path, sep="\t")
    except Exception:
        return None
    if "type" in channels.columns:
        count = int(channels["type"].astype(str).str.upper().eq("EEG").sum())
    else:
        count = int(len(channels))
    return count if count > 0 else None


def scan_sparrkulee_eeg_channel_counts(
    data_root: str | Path,
    tasks: Optional[Sequence[str]] = None,
) -> List[EEGChannelCount]:
    """Return SparrKULee EEG dimensions without decompressing every BDF."""

    root = Path(data_root)
    if not root.exists():
        return []

    task_filter = {str(task).lower() for task in tasks} if tasks is not None else None
    counts: List[EEGChannelCount] = []

    for raw_path in _discover_raw_paths(root):
        entities = _entity_dict(raw_path)
        task = entities.get("task", "").lower()
        if task_filter is not None and task not in task_filter:
            continue

        channels_path = _resolve_bids_sidecar(root, raw_path, "channels.tsv")
        if channels_path is not None:
            n_channels = _count_eeg_rows(channels_path)
            if n_channels is not None:
                counts.append(EEGChannelCount(channels_path, n_channels, "channels.tsv"))
                continue

        # SparrKULee uses a 64-channel BioSemi EEG cap. Recordings with 65 or
        # 73 total BDF channels add Status and, in some sessions, EXG channels.
        counts.append(
            EEGChannelCount(
                raw_path,
                SPARRKULEE_EEG_CHANNELS,
                "SparrKULee BioSemi64 layout",
            )
        )

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
            events_path = _resolve_bids_sidecar(
                self.data_root,
                raw_path,
                "events.tsv",
            )
            channels_path = _resolve_bids_sidecar(
                self.data_root,
                raw_path,
                "channels.tsv",
            )

            recording: Dict[str, Any] = {
                "raw_path": raw_path,
                "channels_path": channels_path,
                "events_path": events_path,
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

    def _inferred_channels_path(
        self,
        original_raw_path: Path,
        readable_raw_path: Path,
    ) -> Path:
        try:
            stat = original_raw_path.stat()
            identity = (
                f"{original_raw_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            )
        except OSError:
            identity = str(original_raw_path.resolve())
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
        sidecar_dir = self.cache_dir / "inferred_sidecars"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / f"{original_raw_path.stem}_{digest}_channels.tsv"
        if sidecar_path.exists():
            return sidecar_path

        raw = mne.io.read_raw_bdf(readable_raw_path, preload=False, verbose=False)
        try:
            names = list(raw.ch_names)
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()

        standard_names = set(mne.channels.make_standard_montage("biosemi64").ch_names)
        acquisition_names = {
            f"{group}{index}"
            for group in ("A", "B")
            for index in range(1, 33)
        }

        eeg_indices = []
        for index, name in enumerate(names):
            stripped = name.strip()
            normalized = stripped
            if len(stripped) >= 2 and stripped[0].upper() in {"A", "B"}:
                try:
                    normalized = f"{stripped[0].upper()}{int(stripped[1:])}"
                except ValueError:
                    pass
            if stripped in standard_names or normalized in acquisition_names:
                eeg_indices.append(index)

        if len(eeg_indices) != SPARRKULEE_EEG_CHANNELS:
            excluded_tokens = (
                "status",
                "trigger",
                "trig",
                "exg",
                "eog",
                "ecg",
                "emg",
                "gsr",
                "resp",
                "plet",
                "temp",
                "audio",
            )
            likely_eeg = [
                index
                for index, name in enumerate(names)
                if not any(token in name.lower() for token in excluded_tokens)
            ]
            if len(likely_eeg) >= SPARRKULEE_EEG_CHANNELS:
                eeg_indices = likely_eeg[:SPARRKULEE_EEG_CHANNELS]
            elif len(names) >= SPARRKULEE_EEG_CHANNELS:
                eeg_indices = list(range(SPARRKULEE_EEG_CHANNELS))
            else:
                raise ValueError(
                    f"SparrKULee recording {original_raw_path} has only "
                    f"{len(names)} total channels; expected at least 64"
                )
            warnings.warn(
                f"Could not identify exactly 64 BioSemi labels in "
                f"{original_raw_path.name}; using the first 64 non-auxiliary "
                "channels from the BDF header.",
                RuntimeWarning,
            )

        eeg_index_set = set(eeg_indices)
        rows = []
        for index, name in enumerate(names):
            lower = name.lower()
            if index in eeg_index_set:
                channel_type = "EEG"
            elif "status" in lower or "trigger" in lower or "trig" in lower:
                channel_type = "TRIG"
            elif "eog" in lower or lower in {"exg1", "exg2"}:
                channel_type = "EOG"
            elif "ecg" in lower or lower in {"exg3", "exg4"}:
                channel_type = "ECG"
            else:
                channel_type = "MISC"
            rows.append({"name": name, "type": channel_type})

        pd.DataFrame(rows).to_csv(sidecar_path, sep="\t", index=False)
        print(
            f"Inferred SparrKULee channel types for {original_raw_path.name}: "
            f"{len(eeg_indices)} EEG / {len(names)} total"
        )
        return sidecar_path

    def _preprocess_readable_recording(
        self,
        recording: Dict[str, Any],
        readable_raw_path: Path,
    ) -> None:
        prepared = dict(recording)
        prepared["raw_path"] = readable_raw_path

        channels_path_value = recording.get("channels_path")
        channels_path = (
            Path(channels_path_value)
            if channels_path_value is not None
            else None
        )
        if channels_path is None or not channels_path.exists():
            channels_path = self._inferred_channels_path(
                Path(recording["raw_path"]),
                readable_raw_path,
            )
        prepared["channels_path"] = channels_path

        super()._preprocess_recording(prepared)

    def _preprocess_recording(self, recording: Dict[str, Any]) -> None:
        raw_path = Path(recording["raw_path"])
        if not raw_path.name.lower().endswith(".bdf.gz"):
            self._preprocess_readable_recording(recording, raw_path)
            return

        materialized_path = self._temporary_bdf_path(raw_path)
        temporary_path = materialized_path.with_suffix(".bdf.tmp")
        try:
            if temporary_path.exists():
                temporary_path.unlink()
            with gzip.open(raw_path, "rb") as source, open(temporary_path, "wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            temporary_path.replace(materialized_path)
            self._preprocess_readable_recording(recording, materialized_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
            if materialized_path.exists():
                materialized_path.unlink()

    @staticmethod
    def _apply_eeg_montage(raw: mne.io.BaseRaw, raw_path: Path) -> None:
        montage = mne.channels.make_standard_montage("biosemi64")
        montage_names = set(montage.ch_names)

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
    "SPARRKULEE_EEG_CHANNELS",
    "SparrKULeeEEGContinuousDataset",
    "scan_sparrkulee_eeg_channel_counts",
]
