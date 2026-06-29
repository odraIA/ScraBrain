"""MEG-XL-compatible word-aligned loader for OpenNeuro ds004408 EEG.

The release stores one TextGrid per audio fragment. Each TextGrid contains both
word and phoneme tiers, so this loader deliberately selects the word tier instead
of treating every TextGrid interval as a word label.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import mne
import numpy as np
import pandas as pd

from .eeg_word_aligned_dataset import (
    _TEXTGRID_INTERVAL_RE,
    _tokenize_text,
    OpenNeuroEEGWordAlignedDataset,
)


DATASET_ID = "ds004408"
DATASET_NAME = "openneuroEEG_ds004408"
TASK_MODE = "listening"
DEFAULT_TASKS: List[str] = ["listening"]
DEFAULT_WORD_TIER_NAMES: Tuple[str, ...] = ("word", "words")
DEFAULT_MONTAGE = "biosemi128"
CACHE_VERSION = "ds004408_word_aligned_v2"

# No canonical participant split is supplied by the dataset. Fine-tuning configs
# should split by shared text segment, not by subject, because all participants
# listened to the same 20 audio fragments.
DEFAULT_VAL_SUBJECTS: List[str] = []
DEFAULT_VAL_SESSIONS: List[str] = []

_TEXTGRID_ITEM_RE = re.compile(
    r"(?ms)^\s*item\s*\[(?P<index>\d+)\]\s*:\s*"
    r"(?P<body>.*?)(?=^\s*item\s*\[\d+\]\s*:|\Z)"
)
_TEXTGRID_NAME_RE = re.compile(
    r'(?m)^\s*name\s*=\s*"(?P<name>(?:""|[^"])*)"\s*$'
)


def _normalise_tier_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


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
            f"Unknown eeg_sensor_type {value!r}; expected one of {sorted(aliases)}"
        )
    return aliases[key]


class OpenNeuroEEGDs004408WordAlignedDataset(OpenNeuroEEGWordAlignedDataset):
    """Expose ds004408 as consecutive word-onset-aligned EEG windows.

    In addition to the shared BIDS EEG behaviour, this wrapper:

    * selects only the word tier from the word+phoneme TextGrid files;
    * applies the BioSemi-128 montage so CrissCross receives real sensor geometry;
    * optionally drops channels marked ``bad`` in each BIDS ``channels.tsv``;
    * can reuse MEG-XL's gradiometer sensor-type embedding (the transfer setup
      used by the Weissbart and Alice EEG experiments).
    """

    def __init__(
        self,
        data_root: str,
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache/ds004408_word_aligned_v2",
        subjects: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[str]] = None,
        tasks: Optional[Sequence[str]] = None,
        l_freq: float = 0.1,
        h_freq: float = 40.0,
        target_sfreq: float = 50.0,
        channel_filter=None,
        max_channel_dim: Optional[int] = 128,
        baseline_duration: float = 0.5,
        clip_range: tuple = (-5, 5),
        tokenizer_name: str = "biocodec",
        allow_missing_word_alignment: bool = False,
        dataset_name: str = DATASET_NAME,
        task_mode: str = TASK_MODE,
        word_tier_names: Optional[Sequence[str]] = None,
        montage_name: str = DEFAULT_MONTAGE,
        drop_bad_channels: bool = True,
        eeg_sensor_type: str = "grad",
        cache_version: str = CACHE_VERSION,
        **kwargs: Any,
    ) -> None:
        self.word_tier_names = tuple(word_tier_names or DEFAULT_WORD_TIER_NAMES)
        self.montage_name = str(montage_name)
        self.drop_bad_channels = bool(drop_bad_channels)
        self.eeg_sensor_type = str(eeg_sensor_type)
        self.eeg_sensor_type_id = _sensor_type_id(eeg_sensor_type)
        self.cache_version = str(cache_version)

        super().__init__(
            data_root=data_root,
            dataset_name=dataset_name,
            task_mode=task_mode,
            segment_length=segment_length,
            subsegment_duration=subsegment_duration,
            words_per_segment=words_per_segment,
            window_onset_offset=window_onset_offset,
            cache_dir=cache_dir,
            subjects=list(subjects) if subjects is not None else None,
            sessions=list(sessions) if sessions is not None else None,
            tasks=list(tasks) if tasks is not None else list(DEFAULT_TASKS),
            l_freq=l_freq,
            h_freq=h_freq,
            target_sfreq=target_sfreq,
            channel_filter=channel_filter,
            max_channel_dim=max_channel_dim,
            baseline_duration=baseline_duration,
            clip_range=clip_range,
            tokenizer_name=tokenizer_name,
            allow_missing_word_alignment=allow_missing_word_alignment,
            **kwargs,
        )

    def _cache_path(self, subject: str, session: str, task: str, run: str) -> Path:
        """Use a ds004408-specific cache key to invalidate older flat-tier caches."""
        base_path = super()._cache_path(subject, session, task, run)
        identity = {
            "cache_version": self.cache_version,
            "word_tier_names": list(self.word_tier_names),
            "montage_name": self.montage_name,
            "drop_bad_channels": self.drop_bad_channels,
            "eeg_sensor_type_id": self.eeg_sensor_type_id,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        return base_path.with_name(f"{base_path.stem}_{self.cache_version}_{digest}.h5")

    def _select_word_tier(
        self, textgrid_path: Path, text: str
    ) -> Tuple[str, List[re.Match[str]]]:
        """Return the configured word tier and its interval matches.

        A flat TextGrid fragment is accepted as a backwards-compatible fallback
        for synthetic smoke tests. Multi-tier files without a recognisable word
        tier fail loudly rather than silently mixing phonemes into word labels.
        """
        tiers: List[Tuple[str, List[re.Match[str]]]] = []
        for item_match in _TEXTGRID_ITEM_RE.finditer(text):
            body = item_match.group("body")
            name_match = _TEXTGRID_NAME_RE.search(body)
            if name_match is None:
                continue
            name = name_match.group("name").replace('""', '"').strip()
            intervals = list(_TEXTGRID_INTERVAL_RE.finditer(body))
            if intervals:
                tiers.append((name, intervals))

        if not tiers:
            flat_intervals = list(_TEXTGRID_INTERVAL_RE.finditer(text))
            if flat_intervals:
                return "flat", flat_intervals
            raise ValueError(f"No interval tiers found in {textgrid_path}")

        preferred = {_normalise_tier_name(name) for name in self.word_tier_names}
        for name, intervals in tiers:
            if _normalise_tier_name(name) in preferred:
                return name, intervals

        for name, intervals in tiers:
            if "word" in _normalise_tier_name(name):
                return name, intervals

        if len(tiers) == 1:
            warnings.warn(
                f"Using the only interval tier {tiers[0][0]!r} in {textgrid_path}; "
                f"configured word tiers are {self.word_tier_names}.",
                RuntimeWarning,
            )
            return tiers[0]

        available = [name for name, _intervals in tiers]
        raise ValueError(
            f"Could not identify a word tier in {textgrid_path}. "
            f"Configured={self.word_tier_names}; available={available}."
        )

    def _build_textgrid_word_events(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        textgrid_path = self._find_textgrid(rec)
        if textgrid_path is None:
            return []
        if not textgrid_path.exists():
            raise FileNotFoundError(
                f"TextGrid sidecar exists but is not materialized: {textgrid_path}. "
                "Run scripts/clone_openneuro_ds004408.sh first."
            )
        if textgrid_path.stat().st_size == 0:
            raise ValueError(f"TextGrid sidecar is empty: {textgrid_path}")

        # ds004408 aligns the beginning of each EEG run with its corresponding
        # audio file. Retain support for an events.tsv offset if one is supplied.
        audio_onset = 0.0
        events_frame = self._read_events(rec)
        if events_frame is not None and "onset" in events_frame.columns and len(events_frame):
            onsets = pd.to_numeric(events_frame["onset"], errors="coerce").dropna()
            if len(onsets):
                audio_onset = float(onsets.min())

        text = textgrid_path.read_text(encoding="utf-8", errors="replace")
        tier_name, interval_matches = self._select_word_tier(textgrid_path, text)
        self.alignment_report.setdefault("textgrid_word_tiers", {})[
            str(textgrid_path)
        ] = tier_name

        events: List[Dict[str, Any]] = []
        for match in interval_matches:
            tokens = _tokenize_text(match.group("text"))
            if not tokens:
                continue
            xmin = float(match.group("xmin"))
            xmax = float(match.group("xmax"))
            duration = max(0.0, xmax - xmin)
            token_duration = duration / len(tokens) if duration > 0 else 0.0
            for token_idx, word in enumerate(tokens):
                onset = audio_onset + xmin + token_idx * token_duration
                events.append(self._word_event(word, onset, token_duration, rec))

        events.sort(key=lambda item: item["window_start"])
        if not events:
            raise ValueError(
                f"Selected tier {tier_name!r} contains no parseable words: {textgrid_path}"
            )
        return events

    @staticmethod
    def _channels_tsv_path(raw_path: Path) -> Path:
        base = raw_path.name.rsplit("_eeg.", 1)[0]
        return raw_path.with_name(f"{base}_channels.tsv")

    def _read_bad_channels(self, raw_path: Path) -> List[str]:
        channels_path = self._channels_tsv_path(raw_path)
        if not channels_path.exists():
            return []
        frame = pd.read_csv(channels_path, sep="\t")
        if "name" not in frame.columns or "status" not in frame.columns:
            return []
        bad_mask = frame["status"].astype(str).str.strip().str.lower().eq("bad")
        return frame.loc[bad_mask, "name"].astype(str).tolist()

    def _read_raw(self, raw_path: Path) -> mne.io.BaseRaw:
        raw = super()._read_raw(raw_path)

        try:
            montage = mne.channels.make_standard_montage(self.montage_name)
            raw.set_montage(
                montage,
                match_case=True,
                on_missing="raise",
                verbose=False,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not apply montage {self.montage_name!r} to {raw_path}: {exc}"
            ) from exc

        if self.drop_bad_channels:
            bad_channels = [
                channel for channel in self._read_bad_channels(raw_path)
                if channel in raw.ch_names
            ]
            if bad_channels:
                raw.drop_channels(bad_channels)

        return raw

    def _ensure_cache(self, rec: Dict[str, Any]) -> Path:
        cache_path = super()._ensure_cache(rec)
        with h5py.File(cache_path, "r+") as h5_file:
            sensor_types = h5_file["sensor_types"]
            expected = np.full(sensor_types.shape, self.eeg_sensor_type_id, dtype=np.int64)
            if not np.array_equal(np.asarray(sensor_types), expected):
                sensor_types[...] = expected
            h5_file.attrs["ds004408_cache_version"] = self.cache_version
            h5_file.attrs["ds004408_word_tiers"] = json.dumps(self.word_tier_names)
            h5_file.attrs["ds004408_montage"] = self.montage_name
            h5_file.attrs["ds004408_drop_bad_channels"] = int(self.drop_bad_channels)
            h5_file.attrs["ds004408_eeg_sensor_type"] = self.eeg_sensor_type
            h5_file.flush()
        return cache_path


# Compatibility aliases for configs/imports that use the dataset id literally.
OpenNeuroEEG_ds004408_WordAlignedDataset = OpenNeuroEEGDs004408WordAlignedDataset
OpenNeuroEEGDS004408WordAlignedDataset = OpenNeuroEEGDs004408WordAlignedDataset


__all__ = [
    "DATASET_ID",
    "DATASET_NAME",
    "TASK_MODE",
    "DEFAULT_TASKS",
    "DEFAULT_WORD_TIER_NAMES",
    "DEFAULT_MONTAGE",
    "CACHE_VERSION",
    "DEFAULT_VAL_SUBJECTS",
    "DEFAULT_VAL_SESSIONS",
    "OpenNeuroEEGDs004408WordAlignedDataset",
    "OpenNeuroEEG_ds004408_WordAlignedDataset",
    "OpenNeuroEEGDS004408WordAlignedDataset",
]
