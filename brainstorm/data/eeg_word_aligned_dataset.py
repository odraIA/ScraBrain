"""Word-aligned EEG datasets for CrissCross word classification."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import _process_single_chunk, is_hdf5_cache_readable


EEG_SENSOR_TYPE_ID = 2
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_BIDS_ENTITY_RE = re.compile(r"(?P<key>[a-zA-Z0-9]+)-(?P<value>[^_]+)")
_TEXTGRID_INTERVAL_RE = re.compile(
    r"xmin\s*=\s*(?P<xmin>[-+0-9.eE]+)\s*"
    r"xmax\s*=\s*(?P<xmax>[-+0-9.eE]+)\s*"
    r"text\s*=\s*\"(?P<text>.*?)\"",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class EEGChannelCount:
    path: Path
    n_channels: int
    method: str


def _read_bids_eeg_channel_count(raw_path: Path) -> Optional[EEGChannelCount]:
    base = raw_path.name.rsplit("_eeg.", 1)[0]
    channels_path = raw_path.with_name(f"{base}_channels.tsv")
    if channels_path.exists():
        try:
            channels = pd.read_csv(channels_path, sep="\t")
            if "type" in channels.columns:
                eeg_rows = channels["type"].astype(str).str.upper().eq("EEG")
                n_channels = int(eeg_rows.sum())
            else:
                n_channels = int(len(channels))
            if n_channels > 0:
                return EEGChannelCount(channels_path, n_channels, "channels.tsv")
        except Exception as exc:
            warnings.warn(f"Could not read EEG channel count from {channels_path}: {exc}", RuntimeWarning)

    suffix = raw_path.suffix.lower()
    try:
        if suffix == ".vhdr":
            raw = mne.io.read_raw_brainvision(raw_path, preload=False, verbose=False)
        elif suffix == ".bdf":
            raw = mne.io.read_raw_bdf(raw_path, preload=False, verbose=False)
        elif suffix == ".edf":
            raw = mne.io.read_raw_edf(raw_path, preload=False, verbose=False)
        else:
            return None
        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False, stim=False, exclude=[])
        n_channels = int(len(picks))
        close = getattr(raw, "close", None)
        if callable(close):
            close()
        if n_channels > 0:
            return EEGChannelCount(raw_path, n_channels, "mne header")
    except Exception as exc:
        warnings.warn(f"Could not read EEG channel count from {raw_path}: {exc}", RuntimeWarning)
    return None


def scan_bids_eeg_channel_counts(
    data_root: str | Path,
    tasks: Optional[Sequence[str]] = None,
) -> List[EEGChannelCount]:
    """Return EEG channel counts for BIDS-like raw EEG recordings under ``data_root``."""
    root = Path(data_root)
    if not root.exists():
        return []
    task_filter = {str(task).lower() for task in tasks} if tasks is not None else None
    counts: List[EEGChannelCount] = []
    for raw_path in sorted(root.rglob("*_eeg.*")):
        if raw_path.suffix.lower() not in BIDSEEGWordAlignedDataset.raw_suffixes:
            continue
        entities = _entity_dict(raw_path)
        task = entities.get("task", "").lower()
        if task == "speechopen":
            continue
        if task_filter is not None and task not in task_filter:
            continue
        count = _read_bids_eeg_channel_count(raw_path)
        if count is not None:
            counts.append(count)
    return counts


def scan_zuco_channel_counts(data_root: str | Path) -> List[EEGChannelCount]:
    """Return EEG channel counts for ZuCo HDF5/MAT recordings."""
    root = Path(data_root)
    if not root.exists():
        return []
    if not (root / "task1 - NR").exists() and (root / "data" / "zuco2" / "task1 - NR").exists():
        root = root / "data" / "zuco2"
    preprocessed_root = root / "task1 - NR" / "Preprocessed"
    if not preprocessed_root.exists():
        return []

    counts: List[EEGChannelCount] = []
    for eeg_path in sorted(preprocessed_root.rglob("*_EEG.mat")):
        try:
            with h5py.File(eeg_path, "r") as h5_file:
                if "EEG/chanlocs/labels" in h5_file:
                    n_channels = int(h5_file["EEG/chanlocs/labels"].shape[0])
                elif "EEG/data" in h5_file:
                    n_channels = int(h5_file["EEG/data"].shape[1])
                else:
                    continue
            if n_channels > 0:
                counts.append(EEGChannelCount(eeg_path, n_channels, "zuco hdf5 metadata"))
        except Exception as exc:
            warnings.warn(f"Could not read ZuCo channel count from {eeg_path}: {exc}", RuntimeWarning)
    return counts


def _clean_token(token: Any) -> Optional[str]:
    text = str(token).strip().strip('"').strip("'").lower()
    if not text or text in {"n/a", "nan", "none", "sp", "<unk>", "unknown"}:
        return None
    match = _TOKEN_RE.search(text)
    return match.group(0).lower() if match else None


def _tokenize_text(text: Any) -> List[str]:
    if text is None:
        return []
    raw = str(text)
    if raw.strip().lower() in {"", "n/a", "nan", "none"}:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(raw)]


def _entity_dict(path: Path) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    for part in path.name.split("_"):
        match = _BIDS_ENTITY_RE.match(part)
        if match:
            entities[match.group("key")] = match.group("value").split(".")[0]
    return entities


def _normalize_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    return {str(value).removeprefix("sub-").removeprefix("ses-").lower() for value in values}


def _safe_attr(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    return str(value)


def _normalize_sensor_xyzdir(sensor_xyzdir: np.ndarray) -> np.ndarray:
    sensor_xyzdir = np.nan_to_num(sensor_xyzdir.astype(np.float32, copy=True), copy=False)
    positions = sensor_xyzdir[:, :3]
    if positions.size == 0 or not np.any(np.isfinite(positions)):
        return sensor_xyzdir
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    scale = float(np.sqrt(3 * np.mean(np.sum(centered ** 2, axis=1))))
    if scale <= 0 or not np.isfinite(scale):
        sensor_xyzdir[:, :3] = 0.0
        return sensor_xyzdir
    sensor_xyzdir[:, :3] = centered / scale
    return sensor_xyzdir


class BIDSEEGWordAlignedDataset(Dataset):
    """BIDS-like EEG word-aligned dataset returning EEG under the ``meg`` key."""

    raw_suffixes = {".vhdr", ".bdf", ".edf"}

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        task_mode: str,
        segment_length: float = 150.0,
        subsegment_duration: float = 3.0,
        words_per_segment: int = 50,
        window_onset_offset: float = -0.5,
        cache_dir: str = "./data/cache/eeg",
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
        tokenizer_name: str = "biocodec",
        allow_missing_word_alignment: bool = True,
        **_: Any,
    ):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.task_mode = task_mode
        self.segment_length = segment_length
        self.subsegment_duration = subsegment_duration
        self.words_per_segment = words_per_segment
        self.window_onset_offset = window_onset_offset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.target_sfreq = target_sfreq
        self.channel_filter = channel_filter
        self.max_channel_dim = max_channel_dim
        self.baseline_duration = baseline_duration
        self.clip_range = clip_range
        self.tokenizer_name = tokenizer_name
        self.allow_missing_word_alignment = allow_missing_word_alignment

        self.subjects = _normalize_filter(subjects)
        self.sessions = _normalize_filter(sessions)
        self.tasks = {str(task).lower() for task in tasks} if tasks is not None else None
        self.alignment_report: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "task_mode": task_mode,
            "skipped_recordings": [],
        }

        if not self.data_root.exists():
            raise FileNotFoundError(
                f"{dataset_name} root does not exist: {self.data_root}. "
                "Rerun scripts/summarize_datasets_info.py if dataset paths changed."
            )

        self.recordings = self._discover_recordings()
        if not self.recordings:
            raise ValueError(
                f"No EEG recordings found in {self.data_root} for dataset={dataset_name}, "
                f"subjects={subjects}, sessions={sessions}, tasks={tasks}"
            )

        self.word_groups: List[List[List[Dict[str, Any]]]] = []
        self._parse_all_recordings()
        self.segment_index = self._build_segment_index()

        if not self.segment_index:
            skipped = len(self.alignment_report["skipped_recordings"])
            raise ValueError(
                f"No word-aligned EEG segments found for {dataset_name}. "
                f"Skipped recordings without usable word alignment: {skipped}. "
                f"Alignment report: {self.alignment_report}"
            )

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        recordings: List[Dict[str, Any]] = []
        raw_paths = [
            path for path in sorted(self.data_root.rglob("*_eeg.*"))
            if path.suffix.lower() in self.raw_suffixes
        ]

        for raw_path in raw_paths:
            entities = _entity_dict(raw_path)
            subject = f"sub-{entities.get('sub', '')}" if "sub" in entities else ""
            session = f"ses-{entities.get('ses', '')}" if "ses" in entities else ""
            task = entities.get("task", "")
            run = entities.get("run", "")

            if self.subjects is not None and entities.get("sub", "").lower() not in self.subjects:
                continue
            if self.sessions is not None and entities.get("ses", "").lower() not in self.sessions:
                continue
            if self.tasks is not None and task.lower() not in self.tasks:
                continue
            if task.lower() == "speechopen":
                continue

            base = raw_path.name.rsplit("_eeg.", 1)[0]
            events_path = raw_path.with_name(f"{base}_events.tsv")
            channels_path = raw_path.with_name(f"{base}_channels.tsv")
            eeg_json_path = raw_path.with_name(f"{base}_eeg.json")

            recordings.append({
                "raw_path": raw_path,
                "events_path": events_path if events_path.exists() else None,
                "channels_path": channels_path if channels_path.exists() else None,
                "eeg_json_path": eeg_json_path if eeg_json_path.exists() else None,
                "subject": subject or "sub-unknown",
                "session": session or "",
                "task": task or "unknown",
                "run": run,
                "entities": entities,
                "cache_path": self._cache_path(subject or "sub-unknown", session, task or "unknown", run),
            })

        return recordings

    def _cache_path(self, subject: str, session: str, task: str, run: str) -> Path:
        identity = {
            "dataset": self.dataset_name,
            "task_mode": self.task_mode,
            "target_sfreq": float(self.target_sfreq),
            "l_freq": float(self.l_freq),
            "h_freq": float(self.h_freq),
            "segment_length": float(self.segment_length),
            "subsegment_duration": float(self.subsegment_duration),
            "window_onset_offset": float(self.window_onset_offset),
            "tokenizer_name": self.tokenizer_name,
            "version": "eeg_word_aligned_v1",
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        safe_parts = [
            self.dataset_name,
            self.task_mode,
            subject or "sub-unknown",
            session or "nosession",
            f"task-{task}",
            f"run-{run or 'none'}",
            f"sfreq-{int(round(self.target_sfreq))}",
            f"tok-{self.tokenizer_name}",
            digest,
        ]
        filename = "_".join(part.replace("/", "-") for part in safe_parts) + ".h5"
        return self.cache_dir / self.dataset_name / self.task_mode / filename

    def _parse_all_recordings(self) -> None:
        for rec_idx, rec in enumerate(self.recordings):
            try:
                word_events = self._build_word_events(rec)
            except Exception as exc:
                self._record_skip(rec, f"word alignment failed: {exc}")
                word_events = []

            groups = self._build_word_groups(word_events)
            self.word_groups.append(groups)
            if not groups:
                self._record_skip(rec, "no complete word groups")
            print(
                f"Recording {rec_idx} ({self.dataset_name} {rec['subject']} "
                f"{rec['session']} {rec['task']} run-{rec['run']}): "
                f"Found {len(groups)} word-aligned segments"
            )

    def _record_skip(self, rec: Dict[str, Any], reason: str) -> None:
        self.alignment_report["skipped_recordings"].append({
            "path": str(rec.get("raw_path", "")),
            "subject": rec.get("subject", ""),
            "session": rec.get("session", ""),
            "task": rec.get("task", ""),
            "run": rec.get("run", ""),
            "reason": reason,
        })

    def _build_word_events(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "eegdash" in self.dataset_name.lower():
            return self._build_eegdash_word_events(rec)

        textgrid_events = self._build_textgrid_word_events(rec)
        if textgrid_events:
            return textgrid_events

        return self._build_generic_events_word_events(rec)

    def _read_events(self, rec: Dict[str, Any]) -> Optional[pd.DataFrame]:
        events_path = rec.get("events_path")
        if events_path is None or not Path(events_path).exists():
            return None
        return pd.read_csv(events_path, sep="\t")

    def _build_eegdash_word_events(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        df = self._read_events(rec)
        if df is None:
            raise FileNotFoundError(f"events.tsv not found for {rec['raw_path']}")

        events: List[Dict[str, Any]] = []
        if "sequence_id" not in df.columns:
            return self._build_generic_events_word_events(rec)

        for sequence_id, group in df.groupby("sequence_id", dropna=False):
            if str(sequence_id).lower() in {"n/a", "nan", "none"}:
                continue

            duration_series = (
                pd.to_numeric(group["duration"], errors="coerce").fillna(0)
                if "duration" in group.columns
                else pd.Series(0, index=group.index)
            )
            trial_type_series = (
                group["trial_type"].astype(str)
                if "trial_type" in group.columns
                else pd.Series("", index=group.index)
            )
            word_rows = group[
                (duration_series > 0)
                & (~trial_type_series.isin(["question", "item_marker", "cloze_marker"]))
            ].sort_values("onset")
            if word_rows.empty:
                continue

            meta = word_rows.iloc[0]
            condition = str(meta.get("condition", "")).lower()
            expected_article = _clean_token(meta.get("expected_article"))
            unexpected_article = _clean_token(meta.get("unexpected_article"))
            expected_noun = _clean_token(meta.get("expected_noun"))
            unexpected_noun = _clean_token(meta.get("unexpected_noun"))
            default_article = unexpected_article if condition == "unexpected" else expected_article
            default_noun = unexpected_noun if condition == "unexpected" else expected_noun
            ending_tokens = _tokenize_text(meta.get("sentence_ending"))
            fallback_tokens = [
                token for token in (default_article, default_noun, *ending_tokens)
                if token
            ]
            if not fallback_tokens:
                self.alignment_report.setdefault("skipped_eegdash_sequences", []).append({
                    "sequence_id": str(sequence_id),
                    "reason": "no recoverable metadata tokens",
                })
                continue

            ending_cursor = 0
            fallback_cursor = 0
            for row in word_rows.itertuples(index=False):
                trial_type = str(getattr(row, "trial_type", "")).lower()
                word = None
                if "article_unexpected" in trial_type:
                    word = unexpected_article
                elif "article_expected" in trial_type:
                    word = expected_article
                elif "article" in trial_type:
                    word = default_article
                elif "noun_unexpected" in trial_type:
                    word = unexpected_noun
                elif "noun_expected" in trial_type:
                    word = expected_noun
                elif "noun" in trial_type:
                    word = default_noun
                elif ending_cursor < len(ending_tokens):
                    word = ending_tokens[ending_cursor]
                    ending_cursor += 1
                else:
                    word = fallback_tokens[min(fallback_cursor, len(fallback_tokens) - 1)]
                    fallback_cursor += 1

                if not word:
                    self.alignment_report.setdefault("skipped_eegdash_sequences", []).append({
                        "sequence_id": str(sequence_id),
                        "trial_type": trial_type,
                        "reason": "could not assign word label",
                    })
                    continue
                onset = float(getattr(row, "onset"))
                duration = float(getattr(row, "duration") or self.subsegment_duration)
                events.append(self._word_event(word, onset, duration, rec, sequence_id=str(sequence_id)))

        events.sort(key=lambda item: item["window_start"])
        return events

    def _build_textgrid_word_events(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        textgrid_path = self._find_textgrid(rec)
        if textgrid_path is None:
            return []

        audio_onset = 0.0
        df = self._read_events(rec)
        if df is not None and "onset" in df.columns and len(df) > 0:
            onsets = pd.to_numeric(df["onset"], errors="coerce").dropna()
            if len(onsets) > 0:
                audio_onset = float(onsets.min())

        events: List[Dict[str, Any]] = []
        text = textgrid_path.read_text(errors="replace")
        for match in _TEXTGRID_INTERVAL_RE.finditer(text):
            tokens = _tokenize_text(match.group("text"))
            if not tokens:
                continue
            xmin = float(match.group("xmin"))
            xmax = float(match.group("xmax"))
            duration = max(0.0, xmax - xmin)
            token_duration = duration / len(tokens) if duration > 0 else 0.0
            for idx, word in enumerate(tokens):
                onset = audio_onset + xmin + idx * token_duration
                events.append(self._word_event(word, onset, token_duration, rec))

        events.sort(key=lambda item: item["window_start"])
        return events

    def _find_textgrid(self, rec: Dict[str, Any]) -> Optional[Path]:
        run = rec.get("run")
        if run:
            candidates = [
                self.data_root / "stimuli" / f"audio{int(run):02d}.TextGrid",
                self.data_root / "stimuli" / f"audio{int(run):02d}.textgrid",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate

        matches = sorted((self.data_root / "stimuli").glob("*.TextGrid")) if (self.data_root / "stimuli").exists() else []
        return matches[0] if len(matches) == 1 else None

    def _build_generic_events_word_events(self, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
        df = self._read_events(rec)
        if df is None:
            raise FileNotFoundError(f"events.tsv not found for {rec['raw_path']}")
        if "onset" not in df.columns:
            raise ValueError(f"events.tsv for {rec['raw_path']} is missing onset column")

        events: List[Dict[str, Any]] = []
        explicit_columns = [
            "word",
            "words",
            "token",
            "text",
            "stimulus",
            "stimuli",
            "transcript",
            "label",
            "value",
        ]

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            if rec.get("task", "").lower() == "listeningcovert":
                phase = " ".join(
                    str(row_dict.get(col, ""))
                    for col in ("phase", "condition", "trial_type", "task")
                ).lower()
                if phase and ("speak" in phase or "speechopen" in phase):
                    continue

            text_value = None
            for column in explicit_columns:
                if column not in row_dict:
                    continue
                candidate = row_dict[column]
                if column == "value" and str(candidate).strip().isdigit():
                    continue
                tokens = _tokenize_text(candidate)
                if tokens:
                    text_value = " ".join(tokens)
                    break
            if text_value is None:
                continue

            tokens = _tokenize_text(text_value)
            onset = float(row_dict["onset"])
            duration_value = row_dict.get("duration", 0.0)
            duration = 0.0 if pd.isna(duration_value) else float(duration_value or 0.0)
            token_duration = duration / len(tokens) if duration > 0 and tokens else 0.0
            for token_idx, word in enumerate(tokens):
                token_onset = onset + token_idx * token_duration
                events.append(self._word_event(word, token_onset, token_duration, rec))

        events.sort(key=lambda item: item["window_start"])
        return events

    def _word_event(
        self,
        word: str,
        onset: float,
        duration: float,
        rec: Dict[str, Any],
        **extra: Any,
    ) -> Dict[str, Any]:
        start = onset + self.window_onset_offset
        end = start + self.subsegment_duration
        event = {
            "word": word,
            "onset": float(onset),
            "duration": float(duration),
            "window_start": float(start),
            "window_end": float(end),
            "dataset_name": self.dataset_name,
            "task_mode": self.task_mode,
            "subject": rec["subject"],
            "session": rec["session"],
            "task": rec["task"],
            "run": rec["run"],
        }
        event.update(extra)
        return event

    def _build_word_groups(self, word_events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        for event in word_events:
            if event["window_start"] < 0:
                continue
            item = dict(event)
            item["subsegment_idx"] = len(current)
            current.append(item)
            if len(current) == self.words_per_segment:
                groups.append(current.copy())
                current = []
        return groups

    def _build_segment_index(self) -> List[Tuple[int, int]]:
        segment_index: List[Tuple[int, int]] = []
        for rec_idx, groups in enumerate(self.word_groups):
            for group_idx in range(len(groups)):
                segment_index.append((rec_idx, group_idx))
        return segment_index

    def __len__(self) -> int:
        return len(self.segment_index)

    def _read_raw(self, raw_path: Path) -> mne.io.BaseRaw:
        suffix = raw_path.suffix.lower()
        if suffix == ".vhdr":
            raw = mne.io.read_raw_brainvision(raw_path, preload=True, verbose=False)
        elif suffix == ".bdf":
            raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose=False)
        elif suffix == ".edf":
            raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported EEG suffix {suffix}: {raw_path}")

        picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, misc=False, stim=False, exclude=[])
        if len(picks) == 0:
            raise ValueError(f"No EEG channels found in {raw_path}")
        raw.pick(picks)

        if self.channel_filter is not None:
            keep = [ch for ch in raw.ch_names if self.channel_filter(ch)]
            raw.pick(keep)

        if self.l_freq is not None or self.h_freq is not None:
            nyquist = raw.info["sfreq"] / 2.0
            h_freq = min(float(self.h_freq), nyquist - 0.5) if self.h_freq is not None else None
            raw.filter(l_freq=self.l_freq, h_freq=h_freq, verbose=False, n_jobs=1)

        if abs(float(raw.info["sfreq"]) - float(self.target_sfreq)) > 0.1:
            old_sfreq = float(raw.info["sfreq"])
            before = raw.n_times
            raw.resample(sfreq=self.target_sfreq, verbose=False, n_jobs=1)
            expected = int(round(before * float(self.target_sfreq) / old_sfreq))
            if raw.n_times <= 0:
                raise ValueError(f"Resampling produced empty recording for {raw_path}")
            if expected > 0 and abs(raw.n_times - expected) > max(2, int(0.02 * expected)):
                warnings.warn(
                    f"Unexpected resampled length for {raw_path}: got {raw.n_times}, "
                    f"expected about {expected}",
                    RuntimeWarning,
                )

        return raw

    def _sensor_xyzdir_from_raw(self, raw: mne.io.BaseRaw) -> np.ndarray:
        rows = []
        for ch in raw.info["chs"]:
            pos = np.asarray(ch["loc"][:3], dtype=np.float32)
            pos = np.nan_to_num(pos, copy=False)
            norm = np.linalg.norm(pos)
            direction = pos / norm if norm > 0 else np.zeros(3, dtype=np.float32)
            rows.append(np.concatenate([pos, direction]))
        return np.asarray(rows, dtype=np.float32)

    def _ensure_cache(self, rec: Dict[str, Any]) -> Path:
        cache_path = Path(rec["cache_path"])
        if is_hdf5_cache_readable(cache_path):
            return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        raw = self._read_raw(rec["raw_path"])
        data = raw.get_data().astype(np.float32, copy=False)
        sensor_xyzdir = self._sensor_xyzdir_from_raw(raw)
        sensor_types = np.full(len(raw.ch_names), EEG_SENSOR_TYPE_ID, dtype=np.int64)
        chunks = (data.shape[0], max(1, int(round(raw.info["sfreq"]))))

        try:
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("data", data=data, chunks=chunks, compression=None)
                f.create_dataset("sensor_xyzdir", data=sensor_xyzdir, compression=None)
                f.create_dataset("sensor_types", data=sensor_types, compression=None)
                f.create_dataset("channel_names", data=[name.encode("utf-8") for name in raw.ch_names], compression=None)
                f.attrs["sample_freq"] = float(raw.info["sfreq"])
                f.attrs["n_channels"] = int(data.shape[0])
                f.attrs["n_samples"] = int(data.shape[1])
                f.attrs["dataset_name"] = self.dataset_name
                f.attrs["task_mode"] = self.task_mode
                f.attrs["tokenizer_name"] = self.tokenizer_name
                f.attrs["preproc_l_freq"] = _safe_attr(self.l_freq)
                f.attrs["preproc_h_freq"] = _safe_attr(self.h_freq)
                f.attrs["preproc_target_sfreq"] = float(self.target_sfreq)
                for key in ("subject", "session", "task", "run"):
                    f.attrs[key] = _safe_attr(rec.get(key))
                f.flush()
            tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return cache_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, group_idx = self.segment_index[idx]
        rec = self.recordings[rec_idx]
        word_group = self.word_groups[rec_idx][group_idx]
        cache_path = self._ensure_cache(rec)

        with h5py.File(cache_path, "r") as h5_file:
            data = h5_file["data"]
            sfreq = float(h5_file.attrs["sample_freq"])
            sensor_xyzdir = np.asarray(h5_file["sensor_xyzdir"], dtype=np.float32)
            sensor_types = np.asarray(h5_file["sensor_types"], dtype=np.int64)
            n_samples = data.shape[1]

            expected_samples = int(round(self.subsegment_duration * sfreq))
            subsegments = []
            for word_info in word_group:
                start = int(round(word_info["window_start"] * sfreq))
                end = start + expected_samples
                if start < 0:
                    pad_left = -start
                    start = 0
                else:
                    pad_left = 0
                pad_right = max(0, end - n_samples)
                end = min(end, n_samples)
                chunk = np.asarray(data[:, start:end], dtype=np.float32)
                if pad_left or pad_right:
                    chunk = np.pad(chunk, ((0, 0), (pad_left, pad_right)))
                if chunk.shape[1] != expected_samples:
                    chunk = np.pad(chunk, ((0, 0), (0, max(0, expected_samples - chunk.shape[1]))))
                    chunk = chunk[:, :expected_samples]
                processed = _process_single_chunk(
                    np.nan_to_num(chunk, copy=False),
                    sensor_types,
                    sfreq,
                    self.baseline_duration,
                    self.clip_range,
                )
                subsegments.append(processed)

        eeg_data = np.concatenate(subsegments, axis=1)
        sensor_xyzdir = _normalize_sensor_xyzdir(sensor_xyzdir)
        sensor_types = sensor_types.copy()

        if self.max_channel_dim is not None:
            original_n_channels = eeg_data.shape[0]
            if original_n_channels > self.max_channel_dim:
                raise ValueError(
                    f"{self.dataset_name} recording has {original_n_channels} channels, "
                    f"but max_channel_dim={self.max_channel_dim}"
                )
            eeg_data = np.pad(eeg_data, ((0, self.max_channel_dim - original_n_channels), (0, 0)))
            sensor_xyzdir = np.pad(sensor_xyzdir, ((0, self.max_channel_dim - sensor_xyzdir.shape[0]), (0, 0)))
            sensor_types = np.pad(sensor_types, (0, self.max_channel_dim - sensor_types.shape[0]))
            sensor_mask = np.zeros(self.max_channel_dim, dtype=np.float32)
            sensor_mask[:original_n_channels] = 1.0
        else:
            sensor_mask = np.ones(eeg_data.shape[0], dtype=np.float32)

        subsegment_boundaries = []
        cursor = 0
        for subsegment in subsegments:
            subsegment_boundaries.append({
                "start_sample": cursor,
                "end_sample": cursor + subsegment.shape[1],
            })
            cursor += subsegment.shape[1]

        words = [word["word"] for word in word_group]
        return {
            "meg": torch.from_numpy(eeg_data).float(),
            "words": words,
            "subsegment_boundaries": subsegment_boundaries,
            "sensor_xyzdir": torch.from_numpy(sensor_xyzdir).float(),
            "sensor_types": torch.from_numpy(sensor_types).int(),
            "sensor_mask": torch.from_numpy(sensor_mask).float(),
            "subject": rec["subject"],
            "session": rec["session"],
            "task": rec["task"],
            "run": rec["run"],
            "dataset_name": self.dataset_name,
            "task_mode": self.task_mode,
            "recording_idx": rec_idx,
            "segment_idx": group_idx,
            "start_time": float(word_group[0]["window_start"]),
            "end_time": float(word_group[-1]["window_end"]),
        }

    def get_segment_words(self, idx: int) -> List[str]:
        rec_idx, group_idx = self.segment_index[idx]
        return [word["word"] for word in self.word_groups[rec_idx][group_idx]]

    def get_segment_metadata(self, idx: int) -> Dict[str, Any]:
        rec_idx, group_idx = self.segment_index[idx]
        rec = self.recordings[rec_idx]
        return {
            "dataset_name": self.dataset_name,
            "task_mode": self.task_mode,
            "subject": rec["subject"],
            "session": rec["session"],
            "task": rec["task"],
            "run": rec["run"],
            "recording_idx": rec_idx,
            "segment_idx": group_idx,
        }

    def get_split_group(self, idx: int, group_kind: str = "auto") -> str:
        meta = self.get_segment_metadata(idx)
        if group_kind == "subject":
            return f"{meta['dataset_name']}:{meta['subject']}"
        if group_kind == "session":
            return f"{meta['dataset_name']}:{meta['subject']}:{meta['session']}"
        if group_kind == "recording":
            return f"{meta['dataset_name']}:{meta['subject']}:{meta['session']}:{meta['task']}:{meta['run']}"
        if group_kind == "sentence":
            return " ".join(self.get_segment_words(idx))
        return f"{meta['dataset_name']}:{meta['subject']}:{meta['session']}:{meta['task']}:{meta['run']}"


class EEGDashWordAlignedDataset(BIDSEEGWordAlignedDataset):
    def __init__(self, *args: Any, dataset_name: str = "eegdash", task_mode: str = "reading", **kwargs: Any):
        super().__init__(*args, dataset_name=dataset_name, task_mode=task_mode, **kwargs)


class OpenNeuroEEGWordAlignedDataset(BIDSEEGWordAlignedDataset):
    def __init__(self, *args: Any, dataset_name: str = "openneuro_eeg", task_mode: str = "listening", **kwargs: Any):
        super().__init__(*args, dataset_name=dataset_name, task_mode=task_mode, **kwargs)


class PooledWordAlignedDataset(Dataset):
    """Concatenate word-aligned datasets while preserving dataset metadata."""

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("PooledWordAlignedDataset requires at least one dataset")
        self.segment_index: List[Tuple[int, int]] = []
        self.word_groups: List[List[List[Dict[str, Any]]]] = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for local_idx in range(len(dataset)):
                self.segment_index.append((dataset_idx, local_idx))
            if hasattr(dataset, "word_groups"):
                self.word_groups.extend(getattr(dataset, "word_groups"))

    def __len__(self) -> int:
        return len(self.segment_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx, local_idx = self.segment_index[idx]
        return self.datasets[dataset_idx][local_idx]

    def get_segment_words(self, idx: int) -> List[str]:
        dataset_idx, local_idx = self.segment_index[idx]
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_segment_words"):
            return dataset.get_segment_words(local_idx)
        sample = dataset[local_idx]
        return list(sample["words"])

    def get_segment_metadata(self, idx: int) -> Dict[str, Any]:
        dataset_idx, local_idx = self.segment_index[idx]
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_segment_metadata"):
            return dataset.get_segment_metadata(local_idx)
        sample = dataset[local_idx]
        return {
            "dataset_name": sample.get("dataset_name", type(dataset).__name__),
            "task_mode": sample.get("task_mode", ""),
            "subject": sample.get("subject", ""),
            "session": sample.get("session", ""),
            "task": sample.get("task", ""),
            "run": sample.get("run", ""),
        }

    def get_split_group(self, idx: int, group_kind: str = "auto") -> str:
        dataset_idx, local_idx = self.segment_index[idx]
        dataset = self.datasets[dataset_idx]
        if hasattr(dataset, "get_split_group"):
            return dataset.get_split_group(local_idx, group_kind)
        meta = self.get_segment_metadata(idx)
        return f"{meta.get('dataset_name')}:{meta.get('subject')}:{meta.get('session')}:{meta.get('task')}:{meta.get('run')}"


def iter_dataset_words(dataset: Dataset) -> Iterable[List[str]]:
    for idx in range(len(dataset)):
        if hasattr(dataset, "get_segment_words"):
            yield dataset.get_segment_words(idx)
        else:
            sample = dataset[idx]
            yield list(sample["words"])
