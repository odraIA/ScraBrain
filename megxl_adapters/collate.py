"""Collate functions for MEG-XL-style variable-channel batches."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .sensor_mask import pad_channels


def _to_epoch_tensor(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 2:
        raise ValueError(f"MEG epoch must have shape (C, T), got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


def _label_to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"label tensor must be scalar, got shape {tuple(value.shape)}")
        return int(value.item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"label array must be scalar, got shape {value.shape}")
        return int(value.reshape(-1)[0])
    return int(value)


def _maybe_tensorize(values: list[Any]) -> torch.Tensor | list[Any]:
    if not values:
        return values
    if any(v is None for v in values):
        return values
    try:
        if all(isinstance(v, (bool, int, np.integer)) for v in values):
            return torch.as_tensor(values, dtype=torch.long)
        if all(isinstance(v, (float, np.floating)) for v in values):
            return torch.as_tensor(values, dtype=torch.float32)
    except Exception:
        pass
    return values


def megxl_collate(batch: list[dict[str, Any]], max_channels: int) -> dict[str, Any]:
    """
    Collate dict samples into a padded MEG batch plus sensor mask.

    Expected sample keys:
        meg, label, dataset_id?, subject_id?, sensor_positions?, sensor_types?
    """
    if not batch:
        raise ValueError("megxl_collate received an empty batch")

    padded_epochs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[int] = []
    dataset_ids: list[Any] = []
    subject_ids: list[Any] = []
    sensor_positions: list[Any] = []
    sensor_types: list[Any] = []
    has_sensor_positions = False
    has_sensor_types = False
    time_len: int | None = None

    for idx, item in enumerate(batch):
        if not isinstance(item, dict):
            raise TypeError(f"Batch item {idx} must be a dict, got {type(item)!r}")
        if "meg" not in item:
            raise KeyError(f"Batch item {idx} is missing required key 'meg'")
        if "label" not in item:
            raise KeyError(f"Batch item {idx} is missing required key 'label'")

        meg = _to_epoch_tensor(item["meg"])
        if time_len is None:
            time_len = int(meg.shape[1])
        elif meg.shape[1] != time_len:
            raise ValueError(
                f"All epochs in a batch must share T. First T={time_len}, "
                f"item {idx} has T={meg.shape[1]}."
            )

        meg_pad, sensor_mask = pad_channels(meg, max_channels)
        padded_epochs.append(meg_pad)
        masks.append(sensor_mask)
        labels.append(_label_to_int(item["label"]))
        dataset_ids.append(item.get("dataset_id"))
        subject_ids.append(item.get("subject_id"))

        if "sensor_positions" in item:
            has_sensor_positions = True
        if "sensor_types" in item:
            has_sensor_types = True
        sensor_positions.append(item.get("sensor_positions"))
        sensor_types.append(item.get("sensor_types"))

    output: dict[str, Any] = {
        "meg": torch.stack(padded_epochs, dim=0),
        "sensor_mask": torch.stack(masks, dim=0).to(dtype=torch.bool),
        "label": torch.as_tensor(labels, dtype=torch.long),
        "dataset_id": _maybe_tensorize(dataset_ids),
        "subject_id": _maybe_tensorize(subject_ids),
    }
    if has_sensor_positions:
        output["sensor_positions"] = sensor_positions
    if has_sensor_types:
        output["sensor_types"] = sensor_types
    return output
