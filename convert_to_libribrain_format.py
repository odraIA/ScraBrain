#!/usr/bin/env python
"""
Convert ALTER/MEG-MASC and Sherlock BIDS data into a LibriBrain-like layout.

The output contains:
  <task>/derivatives/serialised/*_meg.h5  with dataset "data" shaped (C, T)
  <task>/derivatives/events/*_events.tsv  with kind/segment/timemeg/duration
  converted_libribrain_manifest.json      with train/validation/test splits

`train_ddp.py` detects the manifest and can use the converted data via:

  torchrun --nproc_per_node=2 train_ddp.py --data_path ./converted_libribrain_data --task phoneme

For mixed sensor layouts, use:

  --use_sensor_mask true --max_channels <max channels in manifest>
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import h5py
    import mne
    import numpy as np
    from scipy import signal
except ImportError as exc:
    raise SystemExit(
        "Missing conversion dependency. Run this inside the project environment "
        "(for example the Docker/uv environment with mne, h5py, numpy and scipy). "
        f"Original import error: {exc}"
    ) from exc

from megxl_adapters.converted_libribrain import (
    CANONICAL_PHONEMES,
    MANIFEST_NAME,
    normalize_phoneme_label,
)


DEFAULT_PROC_NAME = "converted+ds"
EVENT_COLUMNS = ["kind", "segment", "timemeg", "duration", "sample", "value", "word"]


@dataclass(frozen=True)
class SourceRun:
    dataset: str
    subject: str
    session: str
    task: str
    run: str
    raw_path: Path
    events_path: Path
    channels_path: Path | None = None


@dataclass
class ConvertedRun:
    dataset: str
    subject: str
    session: str
    task: str
    run: str
    h5_path: str
    events_path: str
    source_raw_path: str
    source_events_path: str
    sfreq: float
    source_sfreq: float
    n_channels: int
    n_times: int
    n_events: int
    n_phonemes: int
    n_silences: int

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subject": self.subject,
            "session": self.session,
            "task": self.task,
            "run": self.run,
            "h5_path": self.h5_path,
            "events_path": self.events_path,
            "source_raw_path": self.source_raw_path,
            "source_events_path": self.source_events_path,
            "sfreq": self.sfreq,
            "source_sfreq": self.source_sfreq,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "n_events": self.n_events,
            "n_phonemes": self.n_phonemes,
            "n_silences": self.n_silences,
        }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_runs: list[SourceRun] = []
    if "alter" in args.datasets:
        source_runs.extend(discover_alter_runs(Path(args.alter_root)))
    if "sherlock" in args.datasets:
        source_runs.extend(discover_sherlock_runs(Path(args.sherlock_root)))

    if args.limit_runs is not None:
        source_runs = source_runs[: args.limit_runs]
    if not source_runs:
        raise SystemExit("No source runs found. Check --datasets and input roots.")

    print(f"[INFO] Found {len(source_runs)} source runs")
    converted: list[ConvertedRun] = []
    for idx, run in enumerate(source_runs, start=1):
        print(f"[{idx}/{len(source_runs)}] {run.dataset} {run.subject} {run.session} {run.task}")
        converted.append(convert_run(run, out_dir, args))

    splits = assign_splits(
        converted,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_by=args.split_by,
    )
    write_manifest(out_dir, converted, splits, args)
    print(f"[OK] Wrote {out_dir / MANIFEST_NAME}")
    print(f"[OK] Max channels: {max(run.n_channels for run in converted)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["alter", "sherlock"],
        default=["alter", "sherlock"],
        help="Datasets to convert.",
    )
    parser.add_argument("--alter_root", default="alter_data")
    parser.add_argument("--sherlock_root", default="sherlock_data")
    parser.add_argument("--out_dir", default="converted_libribrain_data")
    parser.add_argument("--target_sfreq", type=float, default=250.0)
    parser.add_argument("--proc_name", default=DEFAULT_PROC_NAME)
    parser.add_argument("--chunk_seconds", type=float, default=20.0)
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--limit_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--split_by", choices=["run", "subject"], default="run")
    parser.add_argument(
        "--alter_sentence_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only ALTER events with trial_type['condition'] == 'sentence'.",
    )
    parser.add_argument("--min_silence", type=float, default=0.05)
    return parser


def discover_alter_runs(root: Path) -> list[SourceRun]:
    runs: list[SourceRun] = []
    for raw_path in sorted(root.glob("sub-*/ses-*/meg/*_task-*_meg.con")):
        match = re.search(r"(sub-[^_/]+)_(ses-[^_/]+)_task-([^_/]+)_meg\.con$", raw_path.name)
        if not match:
            continue
        subject = match.group(1).removeprefix("sub-")
        session = match.group(2).removeprefix("ses-")
        original_task = match.group(3)
        events_path = raw_path.with_name(raw_path.name.replace("_meg.con", "_events.tsv"))
        channels_path = raw_path.with_name(raw_path.name.replace("_meg.con", "_channels.tsv"))
        if not events_path.exists():
            continue
        runs.append(
            SourceRun(
                dataset="alter",
                subject=subject,
                session=session,
                task=f"AlterTask{original_task}",
                run="1",
                raw_path=raw_path,
                events_path=events_path,
                channels_path=channels_path if channels_path.exists() else None,
            )
        )
    return runs


def discover_sherlock_runs(root: Path) -> list[SourceRun]:
    runs: list[SourceRun] = []
    for raw_path in sorted(root.glob("sub-*/ses-*/meg/*_task-compr_meg.ds")):
        match = re.search(r"(sub-[^_/]+)_(ses-[^_/]+)_task-compr_meg\.ds$", raw_path.name)
        if not match:
            continue
        subject = match.group(1).removeprefix("sub-")
        session = match.group(2).removeprefix("ses-")
        events_path = raw_path.with_name(raw_path.name.replace("_meg.ds", "_events.tsv"))
        channels_path = raw_path.with_name(raw_path.name.replace("_meg.ds", "_channels.tsv"))
        if not events_path.exists():
            continue
        runs.append(
            SourceRun(
                dataset="sherlock",
                subject=subject,
                session=session,
                task="SherlockNarrative",
                run="1",
                raw_path=raw_path,
                events_path=events_path,
                channels_path=channels_path if channels_path.exists() else None,
            )
        )
    return runs


def convert_run(run: SourceRun, out_dir: Path, args: argparse.Namespace) -> ConvertedRun:
    serialised_dir = out_dir / run.task / "derivatives" / "serialised"
    events_dir = out_dir / run.task / "derivatives" / "events"
    serialised_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    stem = f"sub-{run.subject}_ses-{run.session}_task-{run.task}_run-{run.run}"
    h5_path = serialised_dir / f"{stem}_proc-{args.proc_name}_meg.h5"
    out_events_path = events_dir / f"{stem}_events.tsv"

    if args.skip_existing and h5_path.exists() and out_events_path.exists():
        with h5py.File(h5_path, "r") as h5_file:
            data = h5_file["data"]
            sfreq = float(h5_file.attrs["sample_frequency"])
            source_sfreq = float(h5_file.attrs.get("source_sample_frequency", sfreq))
            n_channels, n_times = data.shape
        events = read_tsv(out_events_path)
    else:
        raw = read_raw(run)
        source_sfreq = float(raw.info["sfreq"])
        write_h5_from_raw(
            raw=raw,
            h5_path=h5_path,
            target_sfreq=float(args.target_sfreq),
            chunk_seconds=float(args.chunk_seconds),
            compression=None if args.compression == "none" else args.compression,
            source_run=run,
        )
        with h5py.File(h5_path, "r") as h5_file:
            data = h5_file["data"]
            sfreq = float(h5_file.attrs["sample_frequency"])
            n_channels, n_times = data.shape

        if run.dataset == "alter":
            events = convert_alter_events(run.events_path, args.alter_sentence_only)
        elif run.dataset == "sherlock":
            events = convert_sherlock_events(run.events_path)
        else:
            raise ValueError(f"Unknown dataset: {run.dataset}")

        events = add_silence_events(events, total_duration=n_times / sfreq, min_silence=args.min_silence)
        write_events_tsv(out_events_path, events, sfreq=sfreq)

    n_phonemes = sum(1 for row in events if row.get("kind") == "phoneme")
    n_silences = sum(1 for row in events if row.get("kind") == "silence")
    rel_h5 = h5_path.relative_to(out_dir).as_posix()
    rel_events = out_events_path.relative_to(out_dir).as_posix()
    return ConvertedRun(
        dataset=run.dataset,
        subject=run.subject,
        session=run.session,
        task=run.task,
        run=run.run,
        h5_path=rel_h5,
        events_path=rel_events,
        source_raw_path=str(run.raw_path),
        source_events_path=str(run.events_path),
        sfreq=sfreq,
        source_sfreq=source_sfreq,
        n_channels=int(n_channels),
        n_times=int(n_times),
        n_events=len(events),
        n_phonemes=n_phonemes,
        n_silences=n_silences,
    )


def read_raw(run: SourceRun):
    if run.dataset == "alter":
        raw = mne.io.read_raw_kit(str(run.raw_path), preload=False, verbose="ERROR")
    elif run.dataset == "sherlock":
        raw = mne.io.read_raw_ctf(str(run.raw_path), preload=False, verbose="ERROR")
    else:
        raise ValueError(f"Unknown dataset: {run.dataset}")

    picks = mne.pick_types(
        raw.info,
        meg=True,
        eeg=False,
        eog=False,
        ecg=False,
        stim=False,
        misc=False,
        ref_meg=False,
        exclude=[],
    )
    if len(picks) == 0:
        raise RuntimeError(f"No MEG channels found in {run.raw_path}")
    raw.pick(picks)
    return raw


def write_h5_from_raw(
    raw,
    h5_path: Path,
    target_sfreq: float,
    chunk_seconds: float,
    compression: str | None,
    source_run: SourceRun,
) -> None:
    source_sfreq = float(raw.info["sfreq"])
    ratio = target_sfreq / source_sfreq
    n_channels = len(raw.ch_names)
    n_times_out = int(round(raw.n_times * ratio))
    chunk_in = max(1, int(round(chunk_seconds * source_sfreq)))

    sums = np.zeros(n_channels, dtype=np.float64)
    sums_sq = np.zeros(n_channels, dtype=np.float64)
    count = 0

    with h5py.File(h5_path, "w") as h5_file:
        data_ds = h5_file.create_dataset(
            "data",
            shape=(n_channels, n_times_out),
            dtype="float32",
            chunks=(min(n_channels, 64), min(n_times_out, 8192)),
            compression=compression,
        )

        for start in range(0, raw.n_times, chunk_in):
            stop = min(raw.n_times, start + chunk_in)
            out_start = int(round(start * ratio))
            out_stop = int(round(stop * ratio))
            out_len = out_stop - out_start
            if out_len <= 0:
                continue

            chunk = raw.get_data(start=start, stop=stop).astype(np.float32, copy=False)
            if not math.isclose(source_sfreq, target_sfreq):
                chunk = signal.resample(chunk, out_len, axis=1).astype(np.float32, copy=False)
            elif chunk.shape[1] != out_len:
                chunk = chunk[:, :out_len]

            if chunk.shape[1] != out_len:
                chunk = fit_time_length(chunk, out_len)

            data_ds[:, out_start:out_stop] = chunk
            sums += chunk.sum(axis=1, dtype=np.float64)
            sums_sq += np.square(chunk, dtype=np.float64).sum(axis=1)
            count += chunk.shape[1]

        means = sums / max(count, 1)
        variances = np.maximum(sums_sq / max(count, 1) - means**2, 1e-24)
        stds = np.sqrt(variances)

        data_ds.attrs["channel_means"] = means.astype(np.float32)
        data_ds.attrs["channel_stds"] = stds.astype(np.float32)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        data_ds.attrs["channel_names"] = np.asarray(raw.ch_names, dtype=string_dtype)

        h5_file.attrs["sample_frequency"] = float(target_sfreq)
        h5_file.attrs["source_sample_frequency"] = float(source_sfreq)
        h5_file.attrs["dataset"] = source_run.dataset
        h5_file.attrs["subject"] = source_run.subject
        h5_file.attrs["session"] = source_run.session
        h5_file.attrs["task"] = source_run.task
        h5_file.attrs["run"] = source_run.run
        h5_file.attrs["source_raw_path"] = str(source_run.raw_path)


def fit_time_length(data: np.ndarray, expected_len: int) -> np.ndarray:
    if data.shape[1] == expected_len:
        return data
    if data.shape[1] > expected_len:
        return data[:, :expected_len]
    pad = expected_len - data.shape[1]
    return np.pad(data, ((0, 0), (0, pad)), mode="edge")


def convert_alter_events(events_path: Path, sentence_only: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_tsv(events_path):
        trial_type = parse_trial_type(row.get("trial_type"))
        kind = trial_type.get("kind")
        if kind not in {"phoneme", "word"}:
            continue
        if sentence_only and trial_type.get("condition") != "sentence":
            continue
        if str(trial_type.get("pronounced", "1")).lower() in {"0", "0.0", "false"}:
            continue

        onset = to_float(row.get("onset"))
        duration = to_float(row.get("duration"))
        if onset is None or duration is None:
            continue

        if kind == "phoneme":
            segment = normalize_phoneme_label(trial_type.get("phoneme"))
            if segment is None:
                continue
            rows.append(
                {
                    "kind": "phoneme",
                    "segment": segment,
                    "timemeg": onset,
                    "duration": max(duration, 0.0),
                    "value": row.get("value", ""),
                    "word": "",
                }
            )
        elif kind == "word":
            word = str(trial_type.get("word", "")).strip()
            if not word:
                continue
            rows.append(
                {
                    "kind": "word",
                    "segment": "word",
                    "timemeg": onset,
                    "duration": max(duration, 0.0),
                    "value": row.get("value", ""),
                    "word": word,
                }
            )
    return rows


def convert_sherlock_events(events_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_tsv(events_path):
        event_type = str(row.get("type", ""))
        onset = to_float(row.get("onset"))
        duration = to_float(row.get("duration"))
        value = clean_value(row.get("value"))
        if onset is None or duration is None:
            continue

        if event_type.startswith("phoneme_onset"):
            segment = normalize_phoneme_label(value)
            if segment is None:
                if value.lower() in {"sp", "sil", "pau"} and duration > 0:
                    rows.append(silence_row(onset, duration))
                continue
            rows.append(
                {
                    "kind": "phoneme",
                    "segment": segment,
                    "timemeg": onset,
                    "duration": max(duration, 0.0),
                    "value": value,
                    "word": "",
                }
            )
        elif event_type.startswith("word_onset"):
            if value.lower() in {"sp", "sil", "pau"}:
                if duration > 0:
                    rows.append(silence_row(onset, duration))
                continue
            rows.append(
                {
                    "kind": "word",
                    "segment": "word",
                    "timemeg": onset,
                    "duration": max(duration, 0.0),
                    "value": value,
                    "word": value,
                }
            )
    return rows


def add_silence_events(
    events: list[dict[str, Any]],
    total_duration: float,
    min_silence: float,
) -> list[dict[str, Any]]:
    speech_intervals = []
    explicit_silences = []
    for row in events:
        onset = to_float(row.get("timemeg"))
        duration = to_float(row.get("duration"))
        if onset is None or duration is None or duration <= 0:
            continue
        interval = (max(0.0, onset), min(total_duration, onset + duration))
        if interval[1] <= interval[0]:
            continue
        if row.get("kind") == "silence":
            explicit_silences.append(interval)
        elif row.get("kind") in {"phoneme", "word"}:
            speech_intervals.append(interval)

    generated = [silence_row(start, stop - start) for start, stop in invert_intervals(speech_intervals, total_duration, min_silence)]
    all_events = list(events) + generated

    # Drop duplicate silence intervals when Sherlock already provides "sp" spans.
    seen_silences = {(round(start, 4), round(stop, 4)) for start, stop in explicit_silences}
    deduped = []
    for row in all_events:
        if row.get("kind") != "silence":
            deduped.append(row)
            continue
        onset = to_float(row.get("timemeg")) or 0.0
        duration = to_float(row.get("duration")) or 0.0
        key = (round(onset, 4), round(onset + duration, 4))
        if key in seen_silences and row not in events:
            continue
        seen_silences.add(key)
        deduped.append(row)

    deduped.sort(key=lambda item: (float(item.get("timemeg", 0.0)), str(item.get("kind", ""))))
    return deduped


def invert_intervals(
    intervals: Iterable[tuple[float, float]],
    total_duration: float,
    min_silence: float,
) -> list[tuple[float, float]]:
    merged = merge_intervals(intervals)
    silences = []
    cursor = 0.0
    for start, stop in merged:
        if start - cursor >= min_silence:
            silences.append((cursor, start))
        cursor = max(cursor, stop)
    if total_duration - cursor >= min_silence:
        silences.append((cursor, total_duration))
    return silences


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    sorted_intervals = sorted(intervals)
    merged: list[tuple[float, float]] = []
    for start, stop in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def silence_row(onset: float, duration: float) -> dict[str, Any]:
    return {
        "kind": "silence",
        "segment": "sil",
        "timemeg": max(0.0, onset),
        "duration": max(0.0, duration),
        "value": "",
        "word": "",
    }


def write_events_tsv(path: Path, events: list[dict[str, Any]], sfreq: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in events:
            out = dict(row)
            onset = to_float(out.get("timemeg")) or 0.0
            out["sample"] = int(round(onset * sfreq))
            out["timemeg"] = f"{onset:.9f}"
            duration = to_float(out.get("duration"))
            out["duration"] = "n/a" if duration is None else f"{duration:.9f}"
            writer.writerow(out)


def write_manifest(
    out_dir: Path,
    converted: list[ConvertedRun],
    splits: dict[str, list[ConvertedRun]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "format": "converted_libribrain",
        "version": 1,
        "created_by": Path(__file__).name,
        "target_sfreq": float(args.target_sfreq),
        "proc_name": args.proc_name,
        "phoneme_labels": CANONICAL_PHONEMES,
        "n_channels": max(run.n_channels for run in converted),
        "runs": [run.to_manifest_entry() for run in converted],
        "splits": {
            split_name: [run.to_manifest_entry() for run in split_runs]
            for split_name, split_runs in splits.items()
        },
    }
    with (out_dir / MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def assign_splits(
    converted: list[ConvertedRun],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    split_by: str,
) -> dict[str, list[ConvertedRun]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[ConvertedRun]] = {}
    for run in converted:
        key = f"{run.dataset}:{run.subject}" if split_by == "subject" else f"{run.dataset}:{run.subject}:{run.session}:{run.task}:{run.run}"
        groups.setdefault(key, []).append(run)

    keys = list(groups)
    rng.shuffle(keys)
    n_groups = len(keys)
    n_test = max(1, int(round(n_groups * test_fraction))) if n_groups >= 3 else 1
    n_val = max(1, int(round(n_groups * val_fraction))) if n_groups >= 3 else 1
    if n_val + n_test >= n_groups:
        n_val = 1 if n_groups >= 2 else 0
        n_test = 1 if n_groups >= 3 else 0

    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test : n_test + n_val])
    splits = {"train": [], "validation": [], "test": []}
    for key in keys:
        target = "test" if key in test_keys else "validation" if key in val_keys else "train"
        splits[target].extend(groups[key])

    if not splits["train"]:
        splits["train"] = splits["validation"] or splits["test"]
    if not splits["validation"]:
        splits["validation"] = splits["train"][:1]
    if not splits["test"]:
        splits["test"] = splits["validation"][:1]
    return splits


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_trial_type(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().strip('"').strip("'")
        if not text or text.lower() in {"n/a", "nan", "none"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


if __name__ == "__main__":
    main()
