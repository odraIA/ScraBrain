"""Padding and masking helpers for variable-channel MEG/EEG batches."""

from __future__ import annotations

import torch


def pad_channels(x: torch.Tensor, max_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a single epoch from (C, T) to (max_channels, T).

    Returns the padded epoch and a boolean mask with True for real sensors.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"pad_channels expects a torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"pad_channels expects shape (C, T), got {tuple(x.shape)}")
    if max_channels <= 0:
        raise ValueError(f"max_channels must be positive, got {max_channels}")

    channels, time = x.shape
    if channels > max_channels:
        raise ValueError(
            f"Cannot pad {channels} channels to max_channels={max_channels}; "
            "increase --max_channels or drop/remap channels explicitly."
        )

    sensor_mask = torch.zeros(max_channels, dtype=torch.bool, device=x.device)
    sensor_mask[:channels] = True

    if channels == max_channels:
        return x, sensor_mask

    x_pad = x.new_zeros((max_channels, time))
    x_pad[:channels] = x
    return x_pad, sensor_mask


def apply_sensor_mask(
    x: torch.Tensor,
    sensor_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Zero padded channels in raw signals or scalograms.

    Supports:
      - x: (B, C, T), sensor_mask: (B, C) or (C,)
      - x: (B, C, F, T), sensor_mask: (B, C) or (C,)
    """
    if sensor_mask is None:
        return x
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"apply_sensor_mask expects x as torch.Tensor, got {type(x)!r}")
    if not isinstance(sensor_mask, torch.Tensor):
        sensor_mask = torch.as_tensor(sensor_mask, dtype=torch.bool, device=x.device)

    if x.ndim not in (3, 4):
        raise ValueError(
            f"apply_sensor_mask supports x with shape (B, C, T) or (B, C, F, T), "
            f"got {tuple(x.shape)}"
        )

    mask = sensor_mask.to(device=x.device, dtype=torch.bool)
    if mask.ndim == 1:
        if mask.shape[0] != x.shape[1]:
            raise ValueError(
                f"1D sensor_mask has {mask.shape[0]} channels but x has {x.shape[1]}"
            )
        mask = mask.unsqueeze(0).expand(x.shape[0], -1)
    elif mask.ndim == 2:
        if mask.shape != (x.shape[0], x.shape[1]):
            raise ValueError(
                f"sensor_mask shape {tuple(mask.shape)} does not match "
                f"(B, C)=({x.shape[0]}, {x.shape[1]})"
            )
    else:
        raise ValueError(
            f"sensor_mask must have shape (C,) or (B, C), got {tuple(mask.shape)}"
        )

    mask = mask.to(dtype=x.dtype)
    if x.ndim == 3:
        return x * mask[:, :, None]
    return x * mask[:, :, None, None]
