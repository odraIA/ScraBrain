#!/usr/bin/env python3
"""Validate ds004408 word alignment and optionally warm its preprocessed cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brainstorm.data.openneuroEEG_ds004408_word_aligned_dataset import (  # noqa: E402
    CACHE_VERSION,
    OpenNeuroEEGDs004408WordAlignedDataset,
)


def _parse_list(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed: list[str] = []
    for value in values:
        parsed.extend(item.strip() for item in str(value).split(",") if item.strip())
    return parsed or None


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(dataset: OpenNeuroEEGDs004408WordAlignedDataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    word_counter: Counter[str] = Counter()
    recording_segment_counts: Counter[int] = Counter()

    for global_segment_idx, (recording_idx, group_idx) in enumerate(dataset.segment_index):
        recording = dataset.recordings[recording_idx]
        group = dataset.word_groups[recording_idx][group_idx]
        words = [str(event["word"]) for event in group]
        if len(words) != dataset.words_per_segment:
            raise ValueError(
                f"Segment {global_segment_idx} has {len(words)} words; "
                f"expected {dataset.words_per_segment}."
            )
        word_counter.update(words)
        recording_segment_counts[recording_idx] += 1
        rows.append(
            {
                "segment_idx": global_segment_idx,
                "recording_idx": recording_idx,
                "subject": recording["subject"],
                "session": recording["session"],
                "task": recording["task"],
                "run": recording["run"],
                "raw_path": str(recording["raw_path"]),
                "textgrid_path": str(dataset._find_textgrid(recording) or ""),
                "group_idx": group_idx,
                "word_count": len(words),
                "start_time": float(group[0]["window_start"]),
                "end_time": float(group[-1]["window_end"]),
                "sentence_split_key": " ".join(words),
                "words_json": json.dumps(words, ensure_ascii=False),
            }
        )

    used_recordings = sorted(recording_segment_counts)
    selected_tiers = dataset.alignment_report.get("textgrid_word_tiers", {})
    tier_counts = Counter(str(value) for value in selected_tiers.values())
    skipped = list(dataset.alignment_report.get("skipped_recordings", []))

    summary = {
        "generated_at": _iso_now(),
        "dataset_name": dataset.dataset_name,
        "task_mode": dataset.task_mode,
        "cache_version": dataset.cache_version,
        "root": str(dataset.data_root),
        "cache_dir": str(dataset.cache_dir),
        "recordings_discovered": len(dataset.recordings),
        "recordings_with_complete_segments": len(used_recordings),
        "segments": len(rows),
        "words": int(sum(word_counter.values())),
        "unique_words": len(word_counter),
        "words_per_segment": dataset.words_per_segment,
        "subsegment_duration": dataset.subsegment_duration,
        "window_onset_offset": dataset.window_onset_offset,
        "target_sfreq": dataset.target_sfreq,
        "l_freq": dataset.l_freq,
        "h_freq": dataset.h_freq,
        "max_channel_dim": dataset.max_channel_dim,
        "montage_name": dataset.montage_name,
        "drop_bad_channels": dataset.drop_bad_channels,
        "eeg_sensor_type": dataset.eeg_sensor_type,
        "configured_word_tiers": list(dataset.word_tier_names),
        "selected_word_tier_counts": dict(sorted(tier_counts.items())),
        "skipped_recordings": skipped,
        "top_words": word_counter.most_common(50),
    }
    return rows, summary


def warm_recording_cache(
    dataset: OpenNeuroEEGDs004408WordAlignedDataset,
    recording_indices: Sequence[int],
) -> list[str]:
    cache_paths: list[str] = []
    total = len(recording_indices)
    for position, recording_idx in enumerate(recording_indices, start=1):
        recording = dataset.recordings[recording_idx]
        cache_path = dataset._ensure_cache(recording)
        cache_paths.append(str(cache_path))
        print(
            f"[{position:04d}/{total:04d}] cached "
            f"{recording['subject']} run-{recording['run']}: {cache_path}",
            flush=True,
        )
    return cache_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./datasets/OpenNeuroEEG_ds004408")
    parser.add_argument(
        "--cache-dir",
        default=f"./data/cache/{CACHE_VERSION}",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/ds004408_word_aligned",
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=["listening"])
    parser.add_argument("--target-sfreq", type=float, default=50.0)
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--words-per-segment", type=int, default=50)
    parser.add_argument("--subsegment-duration", type=float, default=3.0)
    parser.add_argument("--window-onset-offset", type=float, default=-0.5)
    parser.add_argument("--max-channel-dim", type=int, default=128)
    parser.add_argument("--montage-name", default="biosemi128")
    parser.add_argument("--eeg-sensor-type", default="grad")
    parser.add_argument("--word-tier-names", nargs="*", default=["word", "words"])
    parser.add_argument(
        "--drop-bad-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--warm-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preprocess every recording with at least one complete word segment.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OpenNeuroEEGDs004408WordAlignedDataset(
        data_root=args.root,
        cache_dir=args.cache_dir,
        subjects=_parse_list(args.subjects),
        tasks=_parse_list(args.tasks),
        target_sfreq=args.target_sfreq,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        words_per_segment=args.words_per_segment,
        subsegment_duration=args.subsegment_duration,
        segment_length=args.words_per_segment * args.subsegment_duration,
        window_onset_offset=args.window_onset_offset,
        max_channel_dim=args.max_channel_dim,
        montage_name=args.montage_name,
        drop_bad_channels=args.drop_bad_channels,
        eeg_sensor_type=args.eeg_sensor_type,
        word_tier_names=_parse_list(args.word_tier_names),
        allow_missing_word_alignment=False,
    )

    rows, summary = build_manifest(dataset)
    if not rows:
        raise RuntimeError("No complete ds004408 word-aligned segments were produced.")
    if not summary["selected_word_tier_counts"]:
        raise RuntimeError("No TextGrid word tier was selected.")

    used_recordings = sorted({int(row["recording_idx"]) for row in rows})
    cache_paths: list[str] = []
    if args.warm_cache:
        cache_paths = warm_recording_cache(dataset, used_recordings)
    summary["cache_warmed"] = bool(args.warm_cache)
    summary["cache_files"] = cache_paths

    manifest_csv = output_dir / "word_aligned_manifest.csv"
    summary_json = output_dir / "summary.json"
    alignment_json = output_dir / "alignment_report.json"
    _write_csv(manifest_csv, rows)
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    alignment_json.write_text(
        json.dumps(dataset.alignment_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nds004408 word-aligned preparation completed")
    print(f"Recordings: {summary['recordings_with_complete_segments']}/{summary['recordings_discovered']}")
    print(f"Segments: {summary['segments']}")
    print(f"Words: {summary['words']} ({summary['unique_words']} unique)")
    print(f"Selected tiers: {summary['selected_word_tier_counts']}")
    print(f"Manifest: {manifest_csv}")
    print(f"Summary: {summary_json}")
    print(f"Alignment report: {alignment_json}")


if __name__ == "__main__":
    main()
