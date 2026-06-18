from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SparrKULeeContinuousDatasetTests(unittest.TestCase):
    def test_channel_scanner_reads_compressed_bdf_sidecar(self) -> None:
        try:
            from brainstorm.data.sparrkulee_eeg_continuous_dataset import (
                scan_sparrkulee_eeg_channel_counts,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg_dir = root / "sub-001" / "ses-varyingStories01" / "eeg"
            eeg_dir.mkdir(parents=True)
            raw_path = eeg_dir / (
                "sub-001_ses-varyingStories01_"
                "task-listeningActive_run-01_eeg.bdf.gz"
            )
            raw_path.touch()
            raw_path.with_name(
                "sub-001_ses-varyingStories01_"
                "task-listeningActive_run-01_channels.tsv"
            ).write_text(
                "name\ttype\nA1\tEEG\nA2\tEEG\nEXG1\tEOG\nStatus\tTRIG\n",
                encoding="utf-8",
            )

            counts = scan_sparrkulee_eeg_channel_counts(
                root,
                tasks=["listeningActive"],
            )

        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0].n_channels, 2)
        self.assertEqual(counts[0].method, "channels.tsv")

    def test_biosemi_acquisition_labels_receive_montage(self) -> None:
        try:
            import mne
            import numpy as np
            from brainstorm.data.sparrkulee_eeg_continuous_dataset import (
                SparrKULeeEEGContinuousDataset,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")

        channel_names = [
            f"{group}{index}"
            for group in ("A", "B")
            for index in range(1, 33)
        ]
        info = mne.create_info(channel_names, sfreq=50.0, ch_types="eeg")
        raw = mne.io.RawArray(
            np.zeros((len(channel_names), 10), dtype=np.float64),
            info,
            verbose=False,
        )

        SparrKULeeEEGContinuousDataset._apply_eeg_montage(
            raw,
            Path("synthetic_eeg.bdf"),
        )

        self.assertEqual(raw.ch_names[0], "Fp1")
        self.assertEqual(raw.ch_names[-1], "O2")
        self.assertTrue(
            all(np.linalg.norm(channel["loc"][:3]) > 0 for channel in raw.info["chs"])
        )


if __name__ == "__main__":
    unittest.main()
