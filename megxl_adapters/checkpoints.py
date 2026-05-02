"""Checkpoint helpers shared by MEG-XL pretraining and fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


STATE_DICT_KEYS = ("model_state_dict", "model_state", "state_dict")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = {}
    for key, value in state_dict.items():
        clean_key = key
        while clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        while clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod.") :]
        output[clean_key] = value
    return output


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return strip_module_prefix(checkpoint)
    raise ValueError(
        "Checkpoint does not contain a recognizable state_dict. Expected one of "
        f"{STATE_DICT_KEYS} or a direct state_dict."
    )


def filter_state_dict_for_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    backbone_only: bool = False,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    skipped_mismatch: list[str] = []

    for key, value in state_dict.items():
        if backbone_only and not key.startswith("backbone."):
            unexpected.append(key)
            continue
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_mismatch.append(
                f"{key}: checkpoint{tuple(value.shape)} != model{tuple(model_state[key].shape)}"
            )
            continue
        filtered[key] = value

    return filtered, unexpected, skipped_mismatch


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
    backbone_only: bool = False,
    map_location: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load pretrained model weights.

    With strict=False, keys missing from the current model, unexpected keys, and
    same-name keys with incompatible shapes are skipped and reported instead of
    raising. This is useful when the pretraining head has different classes.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = extract_state_dict(checkpoint)
    model_to_load = unwrap_model(model)

    if strict:
        if backbone_only:
            state_dict = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
        incompatible = model_to_load.load_state_dict(state_dict, strict=True)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
        skipped_mismatch: list[str] = []
        loaded_keys = len(state_dict)
    else:
        filtered, unexpected_keys, skipped_mismatch = filter_state_dict_for_model(
            model_to_load,
            state_dict,
            backbone_only=backbone_only,
        )
        incompatible = model_to_load.load_state_dict(filtered, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(dict.fromkeys(unexpected_keys + list(incompatible.unexpected_keys)))
        loaded_keys = len(filtered)

    report = {
        "path": str(path),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "skipped_mismatch": skipped_mismatch,
        "backbone_only": backbone_only,
        "strict": strict,
    }

    if verbose:
        print(f"[Pretrained] Cargado desde: {path}")
        print(
            f"[Pretrained] strict={strict} backbone_only={backbone_only} | "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
            f"shape_mismatch={len(skipped_mismatch)}"
        )
        if not strict:
            if missing_keys:
                print("[Pretrained] Missing keys:")
                for key in missing_keys:
                    print(f"  - {key}")
            if unexpected_keys:
                print("[Pretrained] Unexpected/skipped keys:")
                for key in unexpected_keys:
                    print(f"  - {key}")
            if skipped_mismatch:
                print("[Pretrained] Shape mismatches skipped:")
                for item in skipped_mismatch:
                    print(f"  - {item}")

    return report
