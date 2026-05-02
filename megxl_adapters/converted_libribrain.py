"""Local LibriBrain-like datasets produced by convert_to_libribrain_format.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


MANIFEST_NAME = "converted_libribrain_manifest.json"

# Stable 39-class ARPAbet inventory used by the training scripts.
CANONICAL_PHONEMES = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "v",
    "w",
    "y",
    "z",
    "zh",
]

PHONEME_TO_ID = {label: idx for idx, label in enumerate(CANONICAL_PHONEMES)}

PHONEME_ALIASES = {
    "ax": "ah",
    "ax-h": "ah",
    "axr": "er",
    "ix": "ih",
    "ux": "uw",
    "hv": "hh",
    "el": "l",
    "em": "m",
    "en": "n",
    "eng": "ng",
    "dx": "d",
}

SILENCE_LABELS = {"", "sp", "sil", "pau", "h#", "oov", "oov_s", "none", "nan", "n/a"}


def manifest_path(data_path: str | Path) -> Path:
    return Path(data_path) / MANIFEST_NAME


def has_converted_manifest(data_path: str | Path) -> bool:
    return manifest_path(data_path).is_file()


def normalize_phoneme_label(raw_label: Any) -> str | None:
    """Normalize ARPAbet labels to the canonical 39-class base inventory."""
    if raw_label is None:
        return None

    label = str(raw_label).strip().strip('"').strip("'").lower()
    label = label.replace("\\ufeff", "")
    if not label:
        return None

    # Sherlock labels often include stress digits: AH0 -> ah.
    label = "".join(ch for ch in label if not ch.isdigit())
    base, sep, suffix = label.partition("_")
    base = base.strip()
    if base in SILENCE_LABELS:
        return None
    if base.endswith("cl"):
        return None

    base = PHONEME_ALIASES.get(base, base)
    if base not in PHONEME_TO_ID:
        return None

    if sep and suffix:
        suffix = suffix.upper()
        if suffix in {"B", "I", "E", "S"}:
            return f"{base}_{suffix}"
    return base


def phoneme_base(label: str) -> str:
    return label.split("_", 1)[0]


class ConvertedLibriBrainDataset(Dataset):
    """
    Dataset compatible with the subset of pnpl's LibriBrain API used here.

    It reads runs serialized as:
      <task>/derivatives/serialised/*_meg.h5 with dataset "data" (C, T)
      <task>/derivatives/events/*_events.tsv

    The root manifest supplies arbitrary run keys, so converted datasets are not
    constrained to pnpl's fixed LibriBrain RUN_KEYS.
    """

    def __init__(
        self,
        data_path: str | Path,
        task: str = "phoneme",
        partition: str = "train",
        tmin: float = 0.0,
        tmax: float | None = None,
        standardize: bool = True,
        clipping_boundary: float | None = 10.0,
        include_info: bool = False,
    ) -> None:
        if task not in {"speech", "phoneme"}:
            raise ValueError(f"Unsupported task: {task!r}")

        self.data_path = Path(data_path)
        self.task = task
        self.partition = partition
        self.tmin = float(tmin)
        self.tmax = float(2.5 if task == "speech" and tmax is None else 0.5 if tmax is None else tmax)
        self.standardize = standardize
        self.clipping_boundary = clipping_boundary
        self.include_info = include_info
        self.open_h5_datasets: dict[tuple[str, str, str, str], h5py.Dataset] = {}
        self.phonemes_sorted = CANONICAL_PHONEMES
        self.phoneme_to_id = PHONEME_TO_ID
        self.id_to_phoneme = CANONICAL_PHONEMES
        self.labels_sorted = CANONICAL_PHONEMES
        self.label_to_id = PHONEME_TO_ID

        manifest_file = manifest_path(self.data_path)
        with manifest_file.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        split_entries = self.manifest.get("splits", {}).get(partition, [])
        if not split_entries:
            raise ValueError(f"No converted runs found for partition {partition!r} in {manifest_file}")

        self.run_entries = [self._resolve_entry(entry) for entry in split_entries]
        self.run_keys = [
            (entry["subject"], entry["session"], entry["task"], entry["run"])
            for entry in self.run_entries
        ]
        self.intended_run_keys = list(self.run_keys)
        self._entry_by_key = {
            (entry["subject"], entry["session"], entry["task"], entry["run"]): entry
            for entry in self.run_entries
        }
        self.n_channels = max(int(entry["n_channels"]) for entry in self.run_entries)
        self.sfreq = float(self.run_entries[0]["sfreq"])
        self.points_per_sample = int(round((self.tmax - self.tmin) * self.sfreq))
        if self.points_per_sample <= 0:
            raise ValueError("tmax must be greater than tmin")

        self._run_stats = self._load_run_stats()
        self.samples: list[tuple[str, str, str, str, float, Any]] = []
        if task == "phoneme":
            self._collect_phoneme_samples()
        else:
            self._collect_speech_samples()

        if not self.samples:
            raise ValueError(f"No {task} samples found in converted partition {partition!r}")

    def _resolve_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(entry)
        resolved["h5_path"] = str(self.data_path / entry["h5_path"])
        resolved["events_path"] = str(self.data_path / entry["events_path"])
        return resolved

    def _load_run_stats(self) -> dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]]:
        stats = {}
        for entry in self.run_entries:
            key = (entry["subject"], entry["session"], entry["task"], entry["run"])
            with h5py.File(entry["h5_path"], "r") as h5_file:
                data = h5_file["data"]
                means = data.attrs.get("channel_means")
                stds = data.attrs.get("channel_stds")
                if means is None or stds is None:
                    means = np.asarray(data[:, :], dtype=np.float32).mean(axis=1)
                    stds = np.asarray(data[:, :], dtype=np.float32).std(axis=1)
                means = np.asarray(means, dtype=np.float32)[:, None]
                stds = np.asarray(stds, dtype=np.float32)[:, None]
                stds = np.where(stds > 1e-12, stds, 1.0).astype(np.float32)
            stats[key] = (means, stds)
        return stats

    def _read_events(self, entry: dict[str, Any]) -> list[dict[str, str]]:
        with open(entry["events_path"], "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def _collect_phoneme_samples(self) -> None:
        for entry in self.run_entries:
            key = (entry["subject"], entry["session"], entry["task"], entry["run"])
            n_times = int(entry["n_times"])
            for row in self._read_events(entry):
                if row.get("kind") != "phoneme":
                    continue
                segment = normalize_phoneme_label(row.get("segment"))
                if segment is None:
                    continue
                onset = _to_float(row.get("timemeg"))
                if onset is None:
                    continue
                start = max(0, int(round((onset + self.tmin) * self.sfreq)))
                if start + self.points_per_sample > n_times:
                    continue
                self.samples.append((*key, onset, segment))

    def _collect_speech_samples(self) -> None:
        for entry in self.run_entries:
            key = (entry["subject"], entry["session"], entry["task"], entry["run"])
            n_times = int(entry["n_times"])
            labels = np.ones(n_times, dtype=np.uint8)

            found_silence = False
            for row in self._read_events(entry):
                if row.get("kind") != "silence":
                    continue
                onset = _to_float(row.get("timemeg"))
                duration = _to_float(row.get("duration"))
                if onset is None or duration is None or duration <= 0:
                    continue
                start = max(0, int(round(onset * self.sfreq)))
                end = min(n_times, start + int(round(duration * self.sfreq)))
                if end > start:
                    labels[start:end] = 0
                    found_silence = True

            if not found_silence:
                labels[:] = 0
                for row in self._read_events(entry):
                    if row.get("kind") not in {"phoneme", "word"}:
                        continue
                    onset = _to_float(row.get("timemeg"))
                    duration = _to_float(row.get("duration"))
                    if onset is None or duration is None or duration <= 0:
                        continue
                    start = max(0, int(round(onset * self.sfreq)))
                    end = min(n_times, start + int(round(duration * self.sfreq)))
                    if end > start:
                        labels[start:end] = 1

            step = self.points_per_sample
            for start in range(0, max(0, n_times - step + 1), step):
                window = labels[start : start + step]
                label = int(window.mean() >= 0.5)
                self.samples.append((*key, start / self.sfreq, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        subject, session, task_name, run, onset, label = self.samples[idx]
        key = (subject, session, task_name, run)
        entry = self._entry_by_key[key]

        if key not in self.open_h5_datasets:
            self.open_h5_datasets[key] = h5py.File(entry["h5_path"], "r")["data"]
        h5_dataset = self.open_h5_datasets[key]

        start = max(0, int(round((float(onset) + self.tmin) * self.sfreq)))
        end = start + self.points_per_sample
        data = np.asarray(h5_dataset[:, start:end], dtype=np.float32)
        if data.shape[1] < self.points_per_sample:
            pad = self.points_per_sample - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")

        if self.standardize:
            means, stds = self._run_stats[key]
            data = (data - means) / stds

        if self.clipping_boundary is not None:
            data = np.clip(data, -self.clipping_boundary, self.clipping_boundary)

        if self.task == "phoneme":
            label_out = self.phoneme_to_id[phoneme_base(str(label))]
        else:
            label_out = int(label)

        if self.include_info:
            info = {
                "dataset": entry.get("dataset", "converted"),
                "subject": subject,
                "session": session,
                "task": task_name,
                "run": run,
                "onset": torch.tensor(float(onset), dtype=torch.float32),
            }
            if self.task == "phoneme":
                info["phoneme_full"] = str(label)
            return [torch.tensor(data, dtype=torch.float32), torch.tensor(label_out), info]

        return [torch.tensor(data, dtype=torch.float32), torch.tensor(label_out)]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().strip('"').strip("'")
        if not text or text.lower() in {"n/a", "nan", "none"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None
