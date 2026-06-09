#!/usr/bin/env python3
"""Create per-dataset text summaries with contents and estimated training hours."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    h5py = None

try:
    import mne  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    mne = None


IGNORED_DIR_NAMES = {
    ".cache",
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    "cache",
    "checkpoints",
    "logs",
}

RAW_SUFFIXES = {
    ".bdf",
    ".cnt",
    ".con",
    ".edf",
    ".egi",
    ".fif",
    ".gdf",
    ".set",
    ".vhdr",
}
SERIALIZED_SUFFIXES = {".h5", ".hdf5"}
MAT_SUFFIXES = {".mat"}

TASK_RE = re.compile(r"(?:^|[_/-])task-([^_./-]+)")
SUB_RE = re.compile(r"(?:^|[_/-])sub-([^_./-]+)")
SES_RE = re.compile(r"(?:^|[_/-])ses-([^_./-]+)")
RUN_RE = re.compile(r"(?:^|[_/-])run-([^_./-]+)")


@dataclass
class Recording:
    path: Path
    kind: str
    task: str
    subject: str
    session: str
    run: str
    duration_s: float | None = None
    sfreq: float | None = None
    n_channels: int | None = None
    n_samples: int | None = None
    method: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan each dataset directory and write one TXT report with contents "
            "and estimated training hours by task."
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Directory containing one subdirectory per dataset. Default: datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets_info"),
        help="Where to write the per-dataset .txt files. Default: datasets_info",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset directory name to process. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="Maximum folder-tree depth included in each TXT. Default: 4",
    )
    parser.add_argument(
        "--max-tree-entries",
        type=int,
        default=2000,
        help="Maximum tree entries printed per dataset. Default: 2000",
    )
    parser.add_argument(
        "--no-read-headers",
        action="store_true",
        help="Skip MNE/HDF5 header reads and only report file contents.",
    )
    parser.add_argument(
        "--include-cache",
        action="store_true",
        help="Do not skip cache/checkpoint/log directories while scanning.",
    )
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024.0 or unit == "PB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def format_duration(seconds: float) -> str:
    seconds_i = int(round(seconds))
    hours = seconds_i // 3600
    minutes = (seconds_i % 3600) // 60
    secs = seconds_i % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def should_skip_dir(path: Path, include_cache: bool) -> bool:
    if include_cache:
        return path.name in {".git", "__pycache__", ".ipynb_checkpoints"}
    return path.name in IGNORED_DIR_NAMES


def walk_dataset(root: Path, include_cache: bool) -> Iterable[tuple[Path, list[str], list[str]]]:
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames if not should_skip_dir(current / d, include_cache)
        )
        yield current, dirnames, sorted(filenames)


def path_entities(path: Path) -> dict[str, str]:
    text = str(path)
    entities = {
        "task": find_entity(text, TASK_RE) or "unknown",
        "subject": find_entity(text, SUB_RE) or "",
        "session": find_entity(text, SES_RE) or "",
        "run": find_entity(text, RUN_RE) or "",
    }

    if entities["task"] == "unknown":
        # LibriBrain stores tasks as task directories such as Sherlock1.
        for part in reversed(path.parts):
            if re.match(r"^(Sherlock|Story|task)[A-Za-z0-9_-]*$", part):
                entities["task"] = part
                break
            if re.match(r"^NR\d+$", part, flags=re.IGNORECASE):
                entities["task"] = "NR"
                entities["session"] = entities["session"] or part.upper()
                break
    if entities["task"] == "unknown" and re.search(r"_NR\d+_EEG\.mat$", path.name, re.IGNORECASE):
        entities["task"] = "NR"

    return entities


def find_entity(text: str, regex: re.Pattern[str]) -> str | None:
    match = regex.search(text)
    return match.group(1) if match else None


def suffix_key(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".fif.gz"):
        return ".fif.gz"
    return path.suffix.lower() or "[no extension]"


def is_raw_signal_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".fif.gz"):
        return True
    return path.suffix.lower() in RAW_SUFFIXES


def is_serialized_signal_file(path: Path) -> bool:
    return path.suffix.lower() in SERIALIZED_SUFFIXES


def is_mat_signal_file(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in MAT_SUFFIXES:
        return False
    return any(token in name for token in ("eeg", "meg", "raw", "data"))


def find_recordings(root: Path, include_cache: bool) -> list[Recording]:
    recordings: list[Recording] = []
    for dirpath, dirnames, filenames in walk_dataset(root, include_cache):
        for dirname in list(dirnames):
            child = dirpath / dirname
            if dirname.lower().endswith(".ds"):
                entities = path_entities(child)
                recordings.append(
                    Recording(
                        path=child,
                        kind="ctf_ds",
                        task=entities["task"],
                        subject=entities["subject"],
                        session=entities["session"],
                        run=entities["run"],
                    )
                )
                dirnames.remove(dirname)

        for filename in filenames:
            path = dirpath / filename
            kind = ""
            if is_raw_signal_file(path):
                kind = "raw"
            elif is_serialized_signal_file(path):
                kind = "hdf5"
            elif is_mat_signal_file(path):
                kind = "mat"
            if not kind:
                continue

            entities = path_entities(path)
            recordings.append(
                Recording(
                    path=path,
                    kind=kind,
                    task=entities["task"],
                    subject=entities["subject"],
                    session=entities["session"],
                    run=entities["run"],
                )
            )
    return recordings


def read_duration(recording: Recording) -> None:
    if recording.kind in {"raw", "ctf_ds"}:
        read_mne_duration(recording)
    elif recording.kind == "hdf5":
        read_hdf5_duration(recording)
    elif recording.kind == "mat":
        read_mat_duration(recording)

    if recording.duration_s is None:
        fallback = duration_from_json_or_events(recording.path)
        if fallback is not None:
            recording.duration_s = fallback
            recording.method = "events/json sidecar"


def read_mne_duration(recording: Recording) -> None:
    if mne is None:
        recording.error = "mne not available"
        return

    path = recording.path
    name = path.name.lower()
    try:
        if recording.kind == "ctf_ds" or name.endswith(".ds"):
            raw = mne.io.read_raw_ctf(str(path), preload=False, verbose=False)
        elif name.endswith(".con"):
            raw = mne.io.read_raw_kit(str(path), preload=False, verbose=False)
        elif name.endswith(".fif") or name.endswith(".fif.gz"):
            raw = mne.io.read_raw_fif(str(path), preload=False, verbose=False)
        elif name.endswith(".edf"):
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
        elif name.endswith(".bdf"):
            raw = mne.io.read_raw_bdf(str(path), preload=False, verbose=False)
        elif name.endswith(".gdf"):
            raw = mne.io.read_raw_gdf(str(path), preload=False, verbose=False)
        elif name.endswith(".vhdr"):
            raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose=False)
        elif name.endswith(".set"):
            raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose=False)
        elif name.endswith(".cnt"):
            raw = mne.io.read_raw_cnt(str(path), preload=False, verbose=False)
        elif name.endswith(".egi"):
            raw = mne.io.read_raw_egi(str(path), preload=False, verbose=False)
        else:
            recording.error = f"unsupported raw suffix: {path.suffix}"
            return

        sfreq = float(raw.info["sfreq"])
        recording.sfreq = sfreq
        recording.n_samples = int(raw.n_times)
        recording.n_channels = len(raw.ch_names)
        recording.duration_s = float(raw.n_times) / sfreq if sfreq else None
        recording.method = "mne header"
        close = getattr(raw, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        recording.error = compact_error(exc)


def read_hdf5_duration(recording: Recording) -> None:
    if h5py is None:
        recording.error = "h5py not available"
        return
    try:
        with h5py.File(recording.path, "r") as h5:
            sfreq = find_hdf5_scalar(h5, ("sample_frequency", "sfreq", "srate", "Fs", "fs"))
            n_samples = find_hdf5_scalar(h5, ("n_samples", "pnts", "samples"))
            n_channels = find_hdf5_scalar(h5, ("n_channels", "nbchan", "channels"))

            if "EEG" in h5:
                eeg = h5["EEG"]
                sfreq = sfreq or find_hdf5_scalar(eeg, ("srate", "sample_frequency", "sfreq"))
                n_samples = n_samples or find_hdf5_scalar(eeg, ("pnts", "n_samples"))
                n_channels = n_channels or find_hdf5_scalar(eeg, ("nbchan", "n_channels"))
                if "data" in eeg and hasattr(eeg["data"], "shape"):
                    shape = tuple(int(x) for x in eeg["data"].shape)
                    n_samples = n_samples or infer_samples_from_shape(shape)
                    n_channels = n_channels or infer_channels_from_shape(shape)

            if "data" in h5 and hasattr(h5["data"], "shape"):
                shape = tuple(int(x) for x in h5["data"].shape)
                n_samples = n_samples or infer_samples_from_shape(shape)
                n_channels = n_channels or infer_channels_from_shape(shape)

            if sfreq is None or n_samples is None:
                recording.error = "missing sample frequency or sample count in HDF5"
                return

            recording.sfreq = float(sfreq)
            recording.n_samples = int(n_samples)
            recording.n_channels = int(n_channels) if n_channels is not None else None
            recording.duration_s = float(n_samples) / float(sfreq)
            recording.method = "hdf5 metadata"
    except Exception as exc:
        recording.error = compact_error(exc)


def read_mat_duration(recording: Recording) -> None:
    if h5py is not None:
        try:
            if h5py.is_hdf5(recording.path):
                read_hdf5_duration(recording)
                if recording.method:
                    recording.method = "mat v7.3/hdf5 metadata"
                return
        except Exception:
            pass

    recording.error = "MAT file is not HDF5/v7.3, duration not inferred without loading it"


def find_hdf5_scalar(obj: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in getattr(obj, "attrs", {}):
            value = scalar_value(obj.attrs[name])
            if value is not None:
                return value
        if name in obj:
            try:
                value = scalar_value(obj[name][()])
            except Exception:
                value = None
            if value is not None:
                return value
    return None


def scalar_value(value: Any) -> float | None:
    try:
        if hasattr(value, "shape") and value.shape == ():
            value = value.item()
        while isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) > 0:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return float(value)
    except Exception:
        return None


def infer_samples_from_shape(shape: tuple[int, ...]) -> int | None:
    if not shape:
        return None
    if len(shape) == 1:
        return shape[0]
    return max(shape)


def infer_channels_from_shape(shape: tuple[int, ...]) -> int | None:
    if len(shape) < 2:
        return None
    return min(shape)


def duration_from_json_or_events(path: Path) -> float | None:
    for json_path in sidecar_json_candidates(path):
        duration = duration_from_json(json_path)
        if duration is not None:
            return duration
    for events_path in events_tsv_candidates(path):
        duration = duration_from_events(events_path)
        if duration is not None:
            return duration
    return None


def sidecar_json_candidates(path: Path) -> list[Path]:
    if path.is_dir() and path.name.endswith(".ds"):
        base = path.with_suffix("")
    else:
        base = path
        for suffix in (".fif.gz", ".hdf5"):
            if path.name.lower().endswith(suffix):
                base = path.with_name(path.name[: -len(suffix)])
                break
        else:
            base = path.with_suffix("")
    candidates = [base.with_suffix(".json")]
    name = base.name
    for signal_suffix in ("_eeg", "_meg", "_ieeg"):
        if name.endswith(signal_suffix):
            candidates.append(base.with_name(name[: -len(signal_suffix)] + signal_suffix + ".json"))
    return [p for p in candidates if p.exists()]


def events_tsv_candidates(path: Path) -> list[Path]:
    parent = path if path.is_dir() else path.parent
    task = find_entity(str(path), TASK_RE)
    stem = path.name
    for raw_suffix in (".fif.gz", ".hdf5", path.suffix):
        if raw_suffix and stem.lower().endswith(raw_suffix):
            stem = stem[: -len(raw_suffix)]
            break
    stem = re.sub(r"_(eeg|meg|ieeg)$", "", stem)
    candidates = [parent / f"{stem}_events.tsv"]
    if task:
        candidates.extend(sorted(parent.glob(f"*task-{task}*_events.tsv")))
    return [p for p in candidates if p.exists()]


def duration_from_json(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return None
    for key in ("RecordingDuration", "Duration", "recording_duration"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def duration_from_events(path: Path) -> float | None:
    try:
        with path.open(newline="", errors="replace") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            max_end = 0.0
            found = False
            for row in reader:
                onset = parse_float(row.get("onset"))
                duration = parse_float(row.get("duration")) or 0.0
                if onset is None:
                    continue
                max_end = max(max_end, onset + duration)
                found = True
            return max_end if found else None
    except Exception:
        return None


def parse_float(value: str | None) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compact_error(exc: Exception) -> str:
    text = str(exc).replace("\n", " ").strip()
    return f"{exc.__class__.__name__}: {text[:240]}"


def collect_content_stats(root: Path, include_cache: bool) -> dict[str, Any]:
    ext_counts: Counter[str] = Counter()
    top_counts: Counter[str] = Counter()
    top_sizes: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    total_files = 0
    total_dirs = 0
    total_size = 0

    for dirpath, dirnames, filenames in walk_dataset(root, include_cache):
        total_dirs += 1
        for filename in filenames:
            path = dirpath / filename
            total_files += 1
            ext_counts[suffix_key(path)] += 1
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            total_size += size

            rel_parts = path.relative_to(root).parts
            top = rel_parts[0] if len(rel_parts) > 1 else "[root files]"
            top_counts[top] += 1
            top_sizes[top] += size

            entities = path_entities(path)
            if entities["subject"]:
                subject_counts["sub-" + entities["subject"]] += 1
            if entities["session"]:
                session_counts["ses-" + entities["session"]] += 1
            if entities["task"] != "unknown":
                task_counts[entities["task"]] += 1

    return {
        "total_dirs": total_dirs,
        "total_files": total_files,
        "total_size": total_size,
        "ext_counts": ext_counts,
        "top_counts": top_counts,
        "top_sizes": top_sizes,
        "subject_counts": subject_counts,
        "session_counts": session_counts,
        "task_counts": task_counts,
    }


def read_dataset_description(root: Path) -> dict[str, Any]:
    path = root / "dataset_description.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        return {"_error": compact_error(exc)}


def count_participants(root: Path) -> int | None:
    path = root / "participants.tsv"
    if not path.exists():
        return None
    try:
        with path.open(newline="", errors="replace") as handle:
            reader = csv.reader(handle, delimiter="\t")
            rows = list(reader)
        return max(0, len(rows) - 1)
    except Exception:
        return None


def tree_lines(root: Path, include_cache: bool, max_depth: int, max_entries: int) -> list[str]:
    lines = ["."]
    emitted = 1

    for dirpath, dirnames, filenames in walk_dataset(root, include_cache):
        rel = dirpath.relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        entries = [(name, True) for name in dirnames] + [(name, False) for name in filenames]
        entries.sort(key=lambda item: (not item[1], item[0].lower()))
        for name, is_dir in entries:
            if emitted >= max_entries:
                lines.append(f"... [tree truncated at {max_entries} entries]")
                return lines
            indent = "  " * (depth + 1)
            suffix = "/" if is_dir else ""
            lines.append(f"{indent}- {name}{suffix}")
            emitted += 1
    return lines


def select_training_recordings(recordings: list[Recording]) -> tuple[list[Recording], str]:
    usable = [rec for rec in recordings if rec.duration_s is not None]
    raw = [rec for rec in usable if rec.kind in {"raw", "ctf_ds"}]
    if raw:
        return raw, "raw recordings with readable duration"

    hdf5 = [rec for rec in usable if rec.kind in {"hdf5", "mat"}]
    if hdf5:
        return hdf5, "serialized HDF5/MAT recordings with readable duration"

    return [], "no recordings with readable duration"


def render_report(
    root: Path,
    stats: dict[str, Any],
    recordings: list[Recording],
    selected: list[Recording],
    selection_note: str,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    description = read_dataset_description(root)
    participants = count_participants(root)

    lines.append(f"Dataset: {root.name}")
    lines.append(f"Path: {root.resolve()}")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    if description:
        lines.append("Dataset metadata")
        if "_error" in description:
            lines.append(f"- dataset_description.json error: {description['_error']}")
        else:
            for key in ("Name", "BIDSVersion", "DatasetType", "License", "Authors"):
                if key in description:
                    lines.append(f"- {key}: {description[key]}")
        if participants is not None:
            lines.append(f"- participants.tsv rows: {participants}")
        lines.append("")

    lines.append("Content summary")
    lines.append(f"- Directories scanned: {stats['total_dirs']}")
    lines.append(f"- Files scanned: {stats['total_files']}")
    lines.append(f"- Total file size: {human_size(stats['total_size'])}")
    if not args.include_cache:
        lines.append(f"- Skipped directory names: {', '.join(sorted(IGNORED_DIR_NAMES))}")
    lines.append("")

    add_counter_section(lines, "Top-level content", stats["top_counts"], stats["top_sizes"])
    add_counter_section(lines, "File extensions", stats["ext_counts"], None)
    add_counter_section(lines, "Detected BIDS-like tasks", stats["task_counts"], None)
    add_counter_section(lines, "Detected BIDS-like subjects", stats["subject_counts"], None, limit=20)
    add_counter_section(lines, "Detected BIDS-like sessions", stats["session_counts"], None, limit=20)

    lines.append("Training-hours estimate")
    lines.append(f"- Selection: {selection_note}")
    lines.append("- Note: all selected recordings are counted as trainable hours; explicit train/val/test splits are not inferred unless encoded as separate files.")
    if selected:
        by_task: dict[str, list[Recording]] = defaultdict(list)
        for rec in selected:
            by_task[rec.task or "unknown"].append(rec)

        total_seconds = sum(rec.duration_s or 0.0 for rec in selected)
        lines.append(f"- Total selected recordings: {len(selected)}")
        lines.append(f"- Total training hours: {total_seconds / 3600.0:.3f} h ({format_duration(total_seconds)})")
        lines.append("")
        lines.append("Hours by task")
        lines.append("task\trecordings\thours\tduration")
        for task, task_recs in sorted(by_task.items()):
            seconds = sum(rec.duration_s or 0.0 for rec in task_recs)
            lines.append(f"{task}\t{len(task_recs)}\t{seconds / 3600.0:.3f}\t{format_duration(seconds)}")
    else:
        lines.append("- Total training hours: unavailable")
    lines.append("")

    if recordings:
        lines.append("Detected recording candidates")
        lines.append("kind\ttask\tsubject\tsession\trun\tduration\tchannels\tsfreq\tmethod\tpath")
        for rec in sorted(recordings, key=lambda r: safe_rel(r.path, root)):
            duration = format_duration(rec.duration_s) if rec.duration_s is not None else "unknown"
            channels = str(rec.n_channels) if rec.n_channels is not None else ""
            sfreq = f"{rec.sfreq:g}" if rec.sfreq is not None else ""
            lines.append(
                "\t".join(
                    [
                        rec.kind,
                        rec.task or "unknown",
                        rec.subject,
                        rec.session,
                        rec.run,
                        duration,
                        channels,
                        sfreq,
                        rec.method,
                        safe_rel(rec.path, root),
                    ]
                )
            )
        lines.append("")

    errors = [rec for rec in recordings if rec.error]
    if errors:
        lines.append("Duration read notes")
        for rec in errors[:100]:
            lines.append(f"- {safe_rel(rec.path, root)}: {rec.error}")
        if len(errors) > 100:
            lines.append(f"- ... {len(errors) - 100} more errors")
        lines.append("")

    lines.append(f"Folder tree (max depth {args.tree_depth})")
    lines.extend(tree_lines(root, args.include_cache, args.tree_depth, args.max_tree_entries))
    lines.append("")

    return "\n".join(lines)


def add_counter_section(
    lines: list[str],
    title: str,
    counter: Counter[str],
    sizes: Counter[str] | None,
    limit: int = 30,
) -> None:
    if not counter:
        return
    lines.append(title)
    for key, count in counter.most_common(limit):
        if sizes is None:
            lines.append(f"- {key}: {count}")
        else:
            lines.append(f"- {key}: {count} files, {human_size(sizes[key])}")
    if len(counter) > limit:
        lines.append(f"- ... {len(counter) - limit} more")
    lines.append("")


def process_dataset(root: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    print(f"[{root.name}] scanning contents", flush=True)
    stats = collect_content_stats(root, args.include_cache)
    recordings = find_recordings(root, args.include_cache)

    if not args.no_read_headers:
        total = len(recordings)
        for idx, recording in enumerate(recordings, start=1):
            print(
                f"[{root.name}] reading duration {idx}/{total}: {safe_rel(recording.path, root)}",
                flush=True,
            )
            read_duration(recording)

    selected, note = select_training_recordings(recordings)
    report = render_report(root, stats, recordings, selected, note, args)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{root.name}.txt"
    output_path.write_text(report, encoding="utf-8")
    print(f"[{root.name}] wrote {output_path}", flush=True)
    return output_path


def main() -> int:
    args = parse_args()
    datasets_root = args.datasets_root.resolve()
    output_dir = args.output_dir.resolve()

    if not datasets_root.exists():
        print(f"datasets root does not exist: {datasets_root}", file=sys.stderr)
        return 2
    if not datasets_root.is_dir():
        print(f"datasets root is not a directory: {datasets_root}", file=sys.stderr)
        return 2

    requested = set(args.dataset or [])
    dataset_dirs = [
        path
        for path in sorted(datasets_root.iterdir())
        if path.is_dir() and not path.name.startswith(".") and (not requested or path.name in requested)
    ]

    missing = requested - {path.name for path in dataset_dirs}
    if missing:
        print(f"requested dataset(s) not found: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2
    if not dataset_dirs:
        print(f"no dataset directories found under {datasets_root}", file=sys.stderr)
        return 2

    written: list[Path] = []
    for dataset_dir in dataset_dirs:
        written.append(process_dataset(dataset_dir, output_dir, args))

    print("")
    print("Reports written:")
    for path in written:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
