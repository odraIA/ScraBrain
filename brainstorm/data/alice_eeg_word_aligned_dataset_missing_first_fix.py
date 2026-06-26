"""Alice EEG loader fix for recordings whose first audio segment is absent.

The original Alice conversion code documents that S26, S34, S35 and S36 have
no event for audio segment 1 and that their remaining event labels are shifted
by one.  This subclass keeps those subjects in the experiment, maps their 11
chronological markers to segments 2--12, and skips only the unavailable words
from segment 1.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import mne
import numpy as np

from .alice_eeg_word_aligned_dataset import AliceEEGWordAlignedDataset


class AliceEEGWordAlignedDatasetMissingFirstFix(AliceEEGWordAlignedDataset):
    """Handle the four Alice recordings with a documented missing first event."""

    MISSING_FIRST_EVENT_SUBJECTS = frozenset({"S26", "S34", "S35", "S36"})

    def _read_segment_onsets(self, raw_path: Path) -> Dict[int, float]:
        raw = mne.io.read_raw_brainvision(str(raw_path), preload=False, verbose=False)
        try:
            annotations = sorted(
                zip(raw.annotations.onset, raw.annotations.description),
                key=lambda item: float(item[0]),
            )
            subject_match = self._SUBJECT_RE.search(raw_path.stem)
            subject = (
                f"S{int(subject_match.group('id')):02d}"
                if subject_match is not None
                else raw_path.stem
            )

            stimulus = [
                (float(onset), str(description))
                for onset, description in annotations
                if "stimulus" in str(description).lower()
            ]
            coded = []
            for onset, description in annotations:
                match = self._TRAILING_TRIGGER_RE.search(str(description))
                if match and 1 <= int(match.group(1)) <= 12:
                    coded.append((float(onset), str(description)))

            def valid_count(markers: List[tuple[float, str]]) -> bool:
                return len(markers) >= 12 or (
                    len(markers) == 11
                    and subject in self.MISSING_FIRST_EVENT_SUBJECTS
                )

            markers = stimulus if valid_count(stimulus) else coded
            if not valid_count(markers):
                expected = (
                    "11 markers for documented segments 2--12"
                    if subject in self.MISSING_FIRST_EVENT_SUBJECTS
                    else "12 markers for segments 1--12"
                )
                raise ValueError(
                    f"Expected {expected} in {raw_path}; found "
                    f"{len(markers)} usable markers."
                )

            if subject in self.MISSING_FIRST_EVENT_SUBJECTS and len(markers) == 11:
                segment_ids = list(range(2, 13))
                selected = markers
                warnings.warn(
                    f"{subject} has the documented missing first Alice event; "
                    "mapping its 11 chronological markers to segments 2--12 and "
                    "omitting segment 1."
                )
            else:
                segment_ids = list(range(1, 13))
                selected = markers[:12]
                if len(markers) > 12:
                    warnings.warn(
                        f"Found {len(markers)} candidate Alice markers in {raw_path}; "
                        "using the first 12 chronological markers."
                    )

            onsets = np.asarray([item[0] for item in selected], dtype=np.float64)
            if not np.all(np.diff(onsets) > 0):
                raise ValueError(
                    f"Alice segment markers are not strictly increasing in {raw_path}."
                )

            result: Dict[int, float] = {}
            for segment_id, onset in zip(segment_ids, onsets):
                lag = self.marker_lag_first if segment_id == 1 else self.marker_lag_other
                result[int(segment_id)] = float(onset + lag)
            return result
        finally:
            close = getattr(raw, "close", None)
            if callable(close):
                close()

    def _make_word_groups(
        self, recording: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        segment_onsets = recording["segment_onsets"]

        for event in self.word_events.itertuples(index=False):
            segment_id = int(event.segment)
            if segment_id not in segment_onsets:
                # Do not bridge a 50-word sequence across unavailable EEG.
                current = []
                continue

            absolute_onset = float(segment_onsets[segment_id]) + float(event.onset)
            start = absolute_onset + self.window_onset_offset
            end = start + self.subsegment_duration
            if start < 0 or end > recording["duration"]:
                current = []
                continue

            current.append(
                {
                    "word": str(event.word),
                    "onset": absolute_onset,
                    "window_start": start,
                    "window_end": end,
                    "segment": segment_id,
                    "order": int(event.order),
                    "subsegment_idx": len(current),
                }
            )
            if len(current) == self.words_per_segment:
                groups.append(current.copy())
                current = []
        return groups
