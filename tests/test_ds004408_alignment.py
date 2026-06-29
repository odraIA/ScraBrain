from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 2
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0.75
            xmax = 1.20
            text = "Hello"
        intervals [2]:
            xmin = 1.20
            xmax = 1.70
            text = "world"
    item [2]:
        class = "IntervalTier"
        name = "phonemes"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0.75
            xmax = 1.20
            text = "HH AH"
        intervals [2]:
            xmin = 1.20
            xmax = 1.70
            text = "W ERLD"
'''


class Ds004408WordAlignmentTests(unittest.TestCase):
    @staticmethod
    def _make_subject(root: Path, subject: str) -> None:
        eeg_dir = root / subject / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        (eeg_dir / f"{subject}_task-listening_run-01_eeg.vhdr").touch()

    def _dataset(self, root: Path):
        try:
            from brainstorm.data.openneuroEEG_ds004408_word_aligned_dataset import (
                OpenNeuroEEGDs004408WordAlignedDataset,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")
        return OpenNeuroEEGDs004408WordAlignedDataset(
            data_root=str(root),
            cache_dir=str(root / "cache"),
            words_per_segment=2,
            subsegment_duration=1.0,
            segment_length=2.0,
            window_onset_offset=-0.5,
            target_sfreq=10.0,
            max_channel_dim=128,
            eeg_sensor_type="eeg",
        )

    def test_word_tier_excludes_phonemes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_subject(root, "sub-001")
            (root / "stimuli").mkdir()
            (root / "stimuli" / "audio01.TextGrid").write_text(TEXTGRID, encoding="utf-8")
            dataset = self._dataset(root)
            self.assertEqual(dataset.get_segment_words(0), ["hello", "world"])
            self.assertEqual(
                set(dataset.alignment_report["textgrid_word_tiers"].values()),
                {"words"},
            )

    def test_sentence_split_key_is_shared_across_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_subject(root, "sub-001")
            self._make_subject(root, "sub-002")
            (root / "stimuli").mkdir()
            (root / "stimuli" / "audio01.TextGrid").write_text(TEXTGRID, encoding="utf-8")
            dataset = self._dataset(root)
            keys = {dataset.get_split_group(idx, "sentence") for idx in range(len(dataset))}
            self.assertEqual(keys, {"hello world"})


if __name__ == "__main__":
    unittest.main()
