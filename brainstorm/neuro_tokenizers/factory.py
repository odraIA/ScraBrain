"""Tokenizer factory for neural signal tokenizers used by CrissCross."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from brainstorm.models.attentional.attn import FeedForward, RMSNorm
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel
from brainstorm.neuro_tokenizers.biocodec.modules.seanet import (
    SEANetDecoder,
    SEANetEncoder,
)


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


class BrainSensorModule(nn.Module):
    """Sensor position/type embedding module from BrainOmni."""

    def __init__(self, n_dim: int):
        super().__init__()
        self.sensor_embedding_layer = nn.Embedding(3, n_dim)
        self.pos_embedding_layer = nn.Sequential(
            nn.Linear(6, n_dim // 2),
            nn.SELU(),
            nn.Linear(n_dim // 2, n_dim),
        )
        self.aggregate_mlp = FeedForward(n_dim, 0.0)
        self.norm = RMSNorm(n_dim)

    def forward(self, pos: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        x = self.pos_embedding_layer(pos)
        x = x + self.sensor_embedding_layer(sensor_type).type_as(x)
        x = x + self.aggregate_mlp(x)
        return self.norm(x)


class ForwardSolution(nn.Module):
    """Project latent neural queries back to physical sensor embeddings."""

    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        if n_dim % n_head != 0:
            raise ValueError(f"n_dim={n_dim} must be divisible by n_head={n_head}")
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.kv = nn.Linear(n_dim, 2 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        sensor_embedding: torch.Tensor,
        neurons: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, _ = sensor_embedding.shape
        kv = self.kv(neurons)
        key, value = torch.split(kv, split_size_or_sections=self.n_dim, dim=-1)
        query = rearrange(sensor_embedding, "B T (H D) -> B H T D", H=self.n_head)
        key = rearrange(key, "B T (H D) -> B H T D", H=self.n_head)
        value = rearrange(value, "B T (H D) -> B H T D", H=self.n_head)
        output = (
            F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=self.dropout,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        return self.proj(output.view(batch, channels, -1))


class BackWardSolution(nn.Module):
    """Map physical sensor features to the tokenizer's latent neural queries."""

    def __init__(self, n_dim: int, n_head: int, dropout: float):
        super().__init__()
        if n_dim % n_head != 0:
            raise ValueError(f"n_dim={n_dim} must be divisible by n_head={n_head}")
        self.n_dim = n_dim
        self.n_head = n_head
        self.dropout = dropout
        self.v = nn.Linear(n_dim, n_dim)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        neuros: torch.Tensor,
        key: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_queries, _ = neuros.shape
        query = rearrange(neuros, "B T (H D) -> B H T D", H=self.n_head)
        key = rearrange(key, "B T (H D) -> B H T D", H=self.n_head)
        value = rearrange(self.v(x), "B T (H D) -> B H T D", H=self.n_head)
        output = (
            F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=self.dropout,
                is_causal=False,
            )
            .transpose(1, 2)
            .contiguous()
        )
        return self.proj(output.view(batch, n_queries, -1))


class BrainTokenizerEncoder(nn.Module):
    """SEANet plus BrainOmni's latent-neuron backward solution."""

    def __init__(
        self,
        n_filters: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        n_dim: int,
        n_head: int,
        dropout: float,
        n_neuro: int,
    ):
        super().__init__()
        self.seanet_encoder = SEANetEncoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )
        self.neuros = nn.Parameter(torch.randn(n_neuro, n_dim))
        self.backwardsolution = BackWardSolution(
            n_dim=n_dim,
            n_head=n_head,
            dropout=dropout,
        )
        self.k_proj = nn.Linear(n_dim, n_dim)

    def forward(
        self,
        x: torch.Tensor,
        sensor_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, channels, n_windows, window_length = x.shape
        x = rearrange(x, "B C N L -> (B C N) 1 L")
        x = self.seanet_encoder(x)
        x = rearrange(
            x,
            "(B C N) D T -> B C (N T) D",
            B=batch,
            C=channels,
            N=n_windows,
        )

        _, _, width, _ = x.shape
        sensor_embedding = rearrange(
            sensor_embedding.unsqueeze(2).repeat(1, 1, width, 1),
            "B C W D -> (B W) C D",
        )
        x = rearrange(x, "B C W D -> (B W) C D")
        neuros = self.neuros.type_as(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.backwardsolution(neuros, self.k_proj(x + sensor_embedding), x)
        x = rearrange(x, "(B N T) C D -> B C (N T) D", B=batch, N=n_windows)
        return rearrange(x, "B C (N T) D -> B C N T D", N=n_windows)


class BrainOmniEuclideanCodebook(nn.Module):
    """Inference-time Euclidean codebook matching BrainOmni checkpoint keys."""

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.register_buffer("inited", torch.ones(1))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", torch.empty(codebook_size, dim))
        self.register_buffer("embed_avg", torch.empty(codebook_size, dim))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = rearrange(x.float(), "... D -> (...) D")
        embed = self.embed.float().t()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        return dist.argmin(dim=-1).view(*shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)


class BrainOmniVectorQuantization(nn.Module):
    """One residual quantizer layer from BrainOmni's EMA RVQ."""

    def __init__(self, dim: int, codebook_dim: int, codebook_size: int):
        super().__init__()
        if codebook_dim != dim:
            self.project_in = nn.Linear(dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        self._codebook = BrainOmniEuclideanCodebook(codebook_dim, codebook_size)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

    @property
    def codebook(self) -> torch.Tensor:
        return self._codebook.embed

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(self.project_in(x))

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self.project_out(self._codebook.decode(indices))

    def forward(self, x: torch.Tensor):
        indices = self.encode(x)
        quantized = self.decode(indices)
        loss = torch.zeros((), device=x.device, dtype=x.dtype)
        return quantized, indices, loss


class BrainOmniRVQ(nn.Module):
    """Residual vector quantizer with BrainOmni's output layout."""

    def __init__(
        self,
        dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList(
            [
                BrainOmniVectorQuantization(dim, codebook_dim, codebook_size)
                for _ in range(num_quantizers)
            ]
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        all_indices = []
        for layer in self.layers:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)

    def forward(self, x: torch.Tensor):
        residual = x
        quantized_out = 0.0
        all_losses = []
        all_indices = []
        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_indices.append(indices)
        return (
            quantized_out,
            torch.stack(all_indices, dim=-1),
            torch.stack(all_losses, dim=-1).mean(),
        )


class BrainQuantizer(nn.Module):
    """BrainOmni quantizer wrapper."""

    def __init__(
        self,
        n_dim: int,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        rotation_trick: bool,
        quantize_optimize_method: str,
    ):
        super().__init__()
        if quantize_optimize_method != "ema":
            raise ValueError(
                "Only BrainOmni EMA quantizer checkpoints are supported; "
                f"got quantize_optimize_method={quantize_optimize_method!r}"
            )
        # rotation_trick affects training only. The frozen adapter only encodes.
        _ = rotation_trick
        self.rvq = BrainOmniRVQ(
            dim=n_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.rvq.encode(F.normalize(x, p=2.0, dim=-1))

    def forward(self, x: torch.Tensor):
        return self.rvq(F.normalize(x, p=2.0, dim=-1))


class BrainTokenizerDecoder(nn.Module):
    """Decoder included so tokenizer checkpoints load completely."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        n_filters: int,
        ratios: list[int],
        kernel_size: int,
        last_kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.forwardsolution = ForwardSolution(n_dim, n_head, dropout)
        self.seanet_decoder = SEANetDecoder(
            channels=1,
            dimension=n_dim,
            n_filters=n_filters,
            ratios=ratios,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        sensor_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, _, n_windows, n_times, dim = x.shape
        x = rearrange(x, "B C N T D -> (B N T) C D")
        sensor_embedding = rearrange(
            sensor_embedding.view(batch, -1, 1, 1, dim).repeat(
                1,
                1,
                n_windows,
                n_times,
                1,
            ),
            "B C N T D -> (B N T) C D",
        )
        x = self.forwardsolution(sensor_embedding, x)
        x = rearrange(
            x,
            "(B N T) C D -> (B C N) D T",
            B=batch,
            N=n_windows,
            T=n_times,
        )
        x = self.seanet_decoder(x)
        return rearrange(x, "(B C N) 1 L -> B C N L", B=batch, N=n_windows)


class BrainTokenizerModel(nn.Module):
    """Functional BrainTokenizer model used inside BrainOmni checkpoints."""

    def __init__(self, **config):
        super().__init__()
        self.window_length = int(config["window_length"])
        self.n_dim = int(config["n_dim"])
        self.sensor_embed = BrainSensorModule(self.n_dim)
        self.encoder = BrainTokenizerEncoder(
            n_filters=int(config["n_filters"]),
            ratios=[int(r) for r in config["ratios"]],
            kernel_size=int(config["kernel_size"]),
            last_kernel_size=int(config["last_kernel_size"]),
            n_dim=self.n_dim,
            n_neuro=int(config["n_neuro"]),
            n_head=int(config["n_head"]),
            dropout=float(config["dropout"]),
        )
        self.quantizer = BrainQuantizer(
            n_dim=self.n_dim,
            codebook_dim=int(config["codebook_dim"]),
            codebook_size=int(config["codebook_size"]),
            num_quantizers=int(config["num_quantizers"]),
            rotation_trick=bool(config.get("rotation_trick", True)),
            quantize_optimize_method=str(config.get("quantize_optimize_method", "ema")),
        )
        self.decoder = BrainTokenizerDecoder(
            n_dim=self.n_dim,
            n_head=int(config["n_head"]),
            n_filters=int(config["n_filters"]),
            ratios=[int(r) for r in config["ratios"]],
            kernel_size=int(config["kernel_size"]),
            last_kernel_size=int(config["last_kernel_size"]),
            dropout=float(config["dropout"]),
        )

    def unfold(self, x: torch.Tensor, overlap_ratio: float = 0.0) -> torch.Tensor:
        if x.shape[-1] < self.window_length:
            x = F.pad(x, pad=(0, self.window_length - x.shape[-1]))
        step = int(self.window_length * (1 - overlap_ratio))
        if step <= 0:
            raise ValueError(f"Invalid overlap_ratio={overlap_ratio}; unfold step is {step}")
        if overlap_ratio > 0.0:
            right_remain = (x.shape[-1] - self.window_length) % step
            if right_remain > 0:
                x = F.pad(x, pad=(0, step - right_remain))
        return x.unfold(dimension=-1, size=self.window_length, step=step)

    @torch.no_grad()
    def tokenize(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        sensor_type: torch.Tensor,
        overlap_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.unfold(x, overlap_ratio=overlap_ratio)
        sensor_embedding = self.sensor_embed(pos, sensor_type)
        feature = self.encoder(x, sensor_embedding)
        feature, indices, _ = self.quantizer(feature)
        feature = rearrange(feature, "B C N T D -> B C (N T) D")
        indices = rearrange(indices, "B C N T Q -> B C (N T) Q")
        return feature, indices


def _convert_legacy_weight_norm_key(key: str) -> str:
    if key.endswith(".weight_g"):
        return f"{key[:-len('.weight_g')]}.parametrizations.weight.original0"
    if key.endswith(".weight_v"):
        return f"{key[:-len('.weight_v')]}.parametrizations.weight.original1"
    return key


def _extract_brain_tokenizer_state_dict(
    checkpoint: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if any(key.startswith("tokenizer.") for key in checkpoint):
        checkpoint = {
            key[len("tokenizer.") :]: value
            for key, value in checkpoint.items()
            if key.startswith("tokenizer.")
        }
    return {
        _convert_legacy_weight_norm_key(key): value
        for key, value in checkpoint.items()
    }


class BrainOmniTokenizerAdapter(NeuroTokenizerAdapter):
    """Adapter for BrainOmni and standalone BrainTokenizer checkpoints."""

    def __init__(
        self,
        tokenizer_name: str,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cpu",
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

        self.overlap_ratio = float(self.config.get("overlap_ratio", 0.0))
        self.model = BrainTokenizerModel(**self.config)

        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"{tokenizer_name} checkpoint must be a state-dict, got {type(checkpoint)}"
            )
        state_dict = _extract_brain_tokenizer_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def encode(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B, 1, T], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(
                "BrainOmniTokenizerAdapter.encode expects one channel per item. "
                "CrissCross passes channels as separate batch items."
            )

        batch = x.shape[0]
        pos = torch.zeros(batch, 1, 6, device=x.device, dtype=x.dtype)
        # BrainOmni uses EEG=0, MAG=1, GRAD=2.
        sensor_type = torch.zeros(batch, 1, device=x.device, dtype=torch.long)

        with torch.no_grad():
            _, indices = self.model.tokenize(
                x,
                pos,
                sensor_type,
                overlap_ratio=self.overlap_ratio,
            )

        # BrainTokenizer emits 16 latent-neuron streams per input sensor. The
        # CrissCross tokenizer contract is one code stream per input channel, so
        # use the first latent stream to keep the output shape [B, Q, T'].
        codes = indices[:, 0, :, :].permute(0, 2, 1).contiguous()
        return [(codes, None)]

    def codebook_embedding(self, q: int) -> torch.Tensor:
        return self.model.quantizer.rvq.layers[q]._codebook.embed


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
    Load a tokenizer by name.

    BioCodec, BrainOmni base/tiny, and the standalone BrainTokenizer checkpoint
    return functional adapters with encode/codebook operations.
    """
    name = str(tokenizer_name or "biocodec").strip().lower()

    if name == "biocodec":
        ckpt = checkpoint_path or DEFAULT_TOKENIZER_ROOT / "biocodec_ckpt.pt"
        return BioCodecTokenizerAdapter(ckpt, device=device)

    if name in {"brainomni_base", "brainomni_tiny", "braintokenizer"}:
        config_path, default_checkpoint = _default_brainomni_paths(name)
        ckpt = Path(checkpoint_path) if checkpoint_path else default_checkpoint
        return BrainOmniTokenizerAdapter(name, config_path, ckpt, device=device)

    raise ValueError(
        f"Unknown tokenizer_name={tokenizer_name!r}. "
        "Expected one of: biocodec, brainomni_base, brainomni_tiny, braintokenizer"
    )
