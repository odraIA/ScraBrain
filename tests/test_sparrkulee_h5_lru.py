from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path


class SparrKULeeH5HandleCacheTests(unittest.TestCase):
    def test_h5_handle_cache_evicts_least_recently_used_file(self) -> None:
        try:
            import h5py
            from brainstorm.data.sparrkulee_eeg_continuous_dataset import (
                SparrKULeeEEGContinuousDataset,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing optional dependency: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(3):
                path = root / f"recording_{index}.h5"
                with h5py.File(path, "w") as h5_file:
                    h5_file.create_dataset("data", data=[index])
                paths.append(path)

            dataset = SparrKULeeEEGContinuousDataset.__new__(
                SparrKULeeEEGContinuousDataset
            )
            dataset.max_open_h5_files = 2
            dataset._file_handles = OrderedDict()
            dataset.source_recordings = [
                {"cache_path": path}
                for path in paths
            ]

            first = dataset._get_h5(0)
            second = dataset._get_h5(1)
            self.assertTrue(first.id.valid)
            self.assertTrue(second.id.valid)

            # Touch file 0 so file 1 becomes the least recently used handle.
            self.assertIs(dataset._get_h5(0), first)
            third = dataset._get_h5(2)

            self.assertTrue(first.id.valid)
            self.assertFalse(second.id.valid)
            self.assertTrue(third.id.valid)
            self.assertEqual(list(dataset._file_handles), [0, 2])

            dataset.close()
            self.assertFalse(first.id.valid)
            self.assertFalse(third.id.valid)


if __name__ == "__main__":
    unittest.main()
