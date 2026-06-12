"""Tokenizer factory for neural signal tokenizers used by CrissCross."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel


DEFAULT_TOKENIZER_ROOT = Path(__file__).resolve().parent


class NeuroTokenizerAdapter(nn.Module):
    """Common tokenizer interface expected by CrissCross."""

    tokenizer_name: str
    downsample_ratio: int
    n_q: int
    vocab_size: int

    def encode(self, x: torch.Tensor):  # pragma: no cover - abstract interface
        raise NotImplementedError

    def codebook_embedding(self, q: int) -> torch.Tensor:  # pragma: no cover - abstract interface
        raise NotImplementedError


class BioCodecTokenizerAdapter(NeuroTokenizerAdapter):
    """Adapter around the existing BioCodec implementation."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        super().__init__()
        self.tokenizer_name = "biocodec"
        self.downsample_ratio = 12
        self.model = BioCodecModel._get_optimized_model()

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"BioCodec tokenizer checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"BioCodec checkpoint {checkpoint_path} is missing 'model_state_dict'"
            )

        state_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            state_dict[key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key] = value

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.n_q = int(self.model.quantizer.n_q)
        self.vocab_size = int(self.model.quantizer.bins)

    def encode(self, x: torch.Tensor):
        return self.model.encode(x)

    def codebook_embedding(self, q: int) -> torch.Tensor:
        return self.model.quantizer.vq.layers[q]._codebook.embed


class BrainOmniTokenizerAdapter(NeuroTokenizerAdapter):
    """
    Metadata-validating placeholder for BrainOmni/BrainTokenizer checkpoints.

    The repository currently contains BrainOmni checkpoint/config files, but no
    model implementation class that can instantiate those weights. This adapter
    exposes config-derived metadata and fails explicitly when model-dependent
    operations are requested.
    """

    _IMPLEMENTATION_MESSAGE = (
        "BrainOmni/BrainTokenizer model code is not available in this repository. "
        "Add the BrainOmni model implementation and wire it into "
        "BrainOmniTokenizerAdapter before using this tokenizer for encoding or "
        "codebook initialization."
    )

    def __init__(
        self,
        tokenizer_name: str,
        config_path: str | Path,
        checkpoint_path: str | Path,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"{tokenizer_name} config not found: {self.config_path}"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"{tokenizer_name} checkpoint not found: {self.checkpoint_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        ratios = self.config.get("ratios")
        if not isinstance(ratios, list) or not ratios:
            raise ValueError(
                f"{self.config_path} must define a non-empty 'ratios' list"
            )

        downsample_ratio = 1
        for ratio in ratios:
            downsample_ratio *= int(ratio)

        self.downsample_ratio = int(downsample_ratio)
        self.n_q = int(
            self.config.get("num_quantizers_used")
            or self.config.get("num_quantizers")
            or 0
        )
        self.vocab_size = int(self.config.get("codebook_size") or 0)

        if self.n_q <= 0:
            raise ValueError(f"{self.config_path} does not define a valid quantizer count")
        if self.vocab_size <= 0:
            raise ValueError(f"{self.config_path} does not define a valid codebook size")

    def encode(self, x: torch.Tensor):
        raise NotImplementedError(self._IMPLEMENTATION_MESSAGE)

    def codebook_embedding(self, q: int) -> torch.Tensor:
        raise NotImplementedError(self._IMPLEMENTATION_MESSAGE)


def _default_brainomni_paths(tokenizer_name: str) -> tuple[Path, Path]:
    mapping = {
        "brainomni_base": (
            DEFAULT_TOKENIZER_ROOT / "base" / "model_cfg.json",
            DEFAULT_TOKENIZER_ROOT / "base" / "BrainOmni.pt",
        ),
        "brainomni_tiny": (
            DEFAULT_TOKENIZER_ROOT / "tiny" / "model_cfg.json",
            DEFAULT_TOKENIZER_ROOT / "tiny" / "BrainOmni.pt",
        ),
        "braintokenizer": (
            DEFAULT_TOKENIZER_ROOT / "braintokenizer" / "model_cfg.json",
            DEFAULT_TOKENIZER_ROOT / "braintokenizer" / "BrainTokenizer.pt",
        ),
    }
    if tokenizer_name not in mapping:
        raise ValueError(f"No default paths registered for tokenizer {tokenizer_name!r}")
    return mapping[tokenizer_name]


def load_neuro_tokenizer(
    tokenizer_name: str = "biocodec",
    checkpoint_path: Optional[str | Path] = None,
    device: str = "cpu",
) -> NeuroTokenizerAdapter:
    """
    Load or validate a tokenizer by name.

    BioCodec returns a functional adapter. BrainOmni variants validate their
    config/checkpoint metadata and expose model dimensions, but fail explicitly
    for encode/codebook operations until the implementation class is present.
    """
    name = str(tokenizer_name or "biocodec").strip().lower()

    if name == "biocodec":
        ckpt = checkpoint_path or DEFAULT_TOKENIZER_ROOT / "biocodec_ckpt.pt"
        return BioCodecTokenizerAdapter(ckpt, device=device)

    if name in {"brainomni_base", "brainomni_tiny", "braintokenizer"}:
        config_path, default_checkpoint = _default_brainomni_paths(name)
        ckpt = Path(checkpoint_path) if checkpoint_path else default_checkpoint
        return BrainOmniTokenizerAdapter(name, config_path, ckpt)

    raise ValueError(
        f"Unknown tokenizer_name={tokenizer_name!r}. "
        "Expected one of: biocodec, brainomni_base, brainomni_tiny, braintokenizer"
    )
