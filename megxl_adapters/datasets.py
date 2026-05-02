"""Dataset wrappers for MEG-XL-style padded-channel training."""

from __future__ import annotations

import bisect
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class LibriBrainRawWrapper(Dataset):
    """
    Wrap a pnpl LibriBrain split and return raw MEG dict samples.

    The output shape is (C, T). Channel padding is intentionally deferred to
    ``megxl_collate`` so this wrapper can coexist with other datasets that have
    different C.
    """

    def __init__(
        self,
        pnpl_dataset,
        preprocessor=None,
        task: str = "phoneme",
        augment: bool = False,
        speech_label_threshold: float = 0.5,
        dataset_id: str = "libribrain",
        subject_id: str = "libribrain_s0",
    ):
        if task not in {"speech", "phoneme"}:
            raise ValueError(f"Unsupported LibriBrain task: {task!r}")
        self.pnpl_dataset = pnpl_dataset
        self.preprocessor = preprocessor
        self.task = task
        self.augment = augment
        self.speech_label_threshold = speech_label_threshold
        self.dataset_id = dataset_id
        self.subject_id = subject_id

    def __len__(self) -> int:
        return len(self.pnpl_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.pnpl_dataset[idx]
        epoch = np.asarray(sample[0], dtype=np.float32)
        label = self._label_to_index(sample[1])

        if self.preprocessor is not None:
            epoch = self.preprocessor(epoch)
        if self.augment:
            epoch = self._augment(epoch)

        return {
            "meg": torch.from_numpy(np.asarray(epoch, dtype=np.float32)),
            "label": int(label),
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
        }

    def _label_to_index(self, raw_label) -> int:
        if isinstance(raw_label, torch.Tensor):
            if raw_label.numel() == 1:
                return int(raw_label.item())
            return int(raw_label.float().mean().item() >= self.speech_label_threshold)

        if isinstance(raw_label, np.ndarray):
            if raw_label.size == 1:
                return int(raw_label.reshape(-1)[0])
            return int(raw_label.astype(np.float32).mean() >= self.speech_label_threshold)

        if isinstance(raw_label, (list, tuple)):
            arr = np.asarray(raw_label)
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
            return int(arr.astype(np.float32).mean() >= self.speech_label_threshold)

        return int(raw_label)

    def _augment(self, epoch: np.ndarray) -> np.ndarray:
        epoch = np.array(epoch, copy=True)
        time_len = epoch.shape[1]

        if np.random.rand() < 0.5:
            max_shift = max(1, int(0.10 * time_len))
            shift = np.random.randint(-max_shift, max_shift + 1)
            epoch = np.roll(epoch, shift, axis=1)

        if np.random.rand() < 0.5:
            epoch = epoch * (1.0 + np.random.uniform(-0.05, 0.05))

        if np.random.rand() < 0.3:
            n_drop = int(0.1 * epoch.shape[0])
            if n_drop > 0:
                drop_idx = np.random.choice(epoch.shape[0], n_drop, replace=False)
                epoch[drop_idx] = 0.0

        return epoch


class MultiDatasetWrapper(Dataset):
    """Concatenate dict-style datasets while ensuring every sample has dataset_id."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        dataset_ids: Sequence[str] | None = None,
    ):
        if not datasets:
            raise ValueError("MultiDatasetWrapper requires at least one dataset")
        if dataset_ids is not None and len(dataset_ids) != len(datasets):
            raise ValueError("dataset_ids must match the number of datasets")

        self.datasets = list(datasets)
        self.dataset_ids = (
            list(dataset_ids)
            if dataset_ids is not None
            else [f"dataset_{idx}" for idx in range(len(self.datasets))]
        )
        lengths = [len(ds) for ds in self.datasets]
        self.cumulative_lengths = np.cumsum(lengths).tolist()

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        previous = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        local_idx = idx - previous
        item = self.datasets[dataset_idx][local_idx]

        if isinstance(item, dict):
            output = dict(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            output = {"meg": item[0], "label": item[1]}
        else:
            raise TypeError(
                "MultiDatasetWrapper expects child datasets to return dicts or "
                f"(meg, label) tuples, got {type(item)!r}"
            )

        output.setdefault("dataset_id", self.dataset_ids[dataset_idx])
        output.setdefault("subject_id", f"{self.dataset_ids[dataset_idx]}_unknown_subject")
        return output


# TODO: Add dataset-specific wrappers for Armeni, MEG-MASC, Broderick, and
# other MEG/EEG corpora when their local/pnpl APIs are available in this repo.
