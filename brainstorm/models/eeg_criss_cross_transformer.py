"""EEG-specific Criss-Cross Transformer with explicit validity masks.

This module preserves the parameter names of the original MEG-XL transformer so
compatible MEG checkpoints can initialize the RVQ projector, spatial encoders,
attention layers, MEG coil-type embeddings and output head.

Differences from the legacy module:
- sensor padding is excluded from spatial and temporal attention;
- time padding is excluded from temporal attention;
- reconstruction blocks are sampled only from target-valid time ranges;
- EEG/MEG modality and MEG coil type are represented separately;
- MEG orientation embeddings are gated off for EEG electrodes.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from brainstorm.models.attentional.attn import (
    FeedForward,
    RMSNorm,
    RotaryEmbedding,
)
from brainstorm.models.criss_cross_transformer import (
    CrissCrossTransformerModule as LegacyCrissCrossTransformerModule,
)


GRAD_SENSOR_TYPE_ID = 0
MAG_SENSOR_TYPE_ID = 1
EEG_SENSOR_TYPE_ID = 2
MEG_MODALITY_ID = 0
EEG_MODALITY_ID = 1


class MaskedSelfAttention(nn.Module):
    """Self-attention with a correctly broadcast boolean key mask."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        dropout: float,
        causal: bool = False,
        rope: bool = False,
    ) -> None:
        super().__init__()
        if n_dim % n_head != 0:
            raise ValueError(
                f"n_dim={n_dim} must be divisible by n_head={n_head}"
            )
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.causal = causal
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope = rope
        self.rope_embedding_layer = (
            RotaryEmbedding(n_dim=n_dim, init_seq_len=240)
            if rope
            else nn.Identity()
        )

    @staticmethod
    def _attention_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = mask.bool()
        if mask.ndim == 2:
            # Avoid all-masked rows in SDPA. These rows correspond to invalid
            # sensor/time queries and are zeroed by the enclosing block.
            empty_rows = ~mask.any(dim=-1)
            if bool(empty_rows.any()):
                mask = mask.clone()
                mask[empty_rows, 0] = True
            # [B, S] -> [B, 1, 1, S], broadcast over heads and queries.
            return mask[:, None, None, :]
        if mask.ndim == 3:
            # [B, L, S] -> [B, 1, L, S].
            return mask[:, None, :, :]
        if mask.ndim == 4:
            return mask
        raise ValueError(
            "attention mask must have shape [B,S], [B,L,S] or [B,H,L,S], "
            f"got {tuple(mask.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(
            qkv,
            split_size_or_sections=self.n_dim,
            dim=-1,
        )

        if self.rope:
            q = q.view(batch, seq_len, self.n_head, -1)
            k = k.view(batch, seq_len, self.n_head, -1)
            q, k = self.rope_embedding_layer(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = rearrange(q, "B T (H D) -> B H T D", H=self.n_head)
            k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)

        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)
        output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=self._attention_mask(mask),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch,
            seq_len,
            self.n_dim,
        )
        return self.proj(output)


class MaskedSpatialTemporalAttentionBlock(nn.Module):
    """Criss-cross block that never lets invalid tokens influence valid ones."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.time_attn = MaskedSelfAttention(
            n_dim // 2,
            n_head // 2,
            dropout,
            causal=causal,
            rope=True,
        )
        self.spatial_attn = MaskedSelfAttention(
            n_dim // 2,
            n_head // 2,
            dropout,
            causal=False,
            rope=False,
        )
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)

    @staticmethod
    def _valid_grid(
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor],
        time_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, channels, timesteps, _ = x.shape
        if sensor_mask is None:
            sensor_mask = torch.ones(
                batch,
                channels,
                dtype=torch.bool,
                device=x.device,
            )
        if time_mask is None:
            time_mask = torch.ones(
                batch,
                timesteps,
                dtype=torch.bool,
                device=x.device,
            )
        return sensor_mask.bool().unsqueeze(-1) & time_mask.bool().unsqueeze(1)

    def _attn_operator(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, _, dim = x.shape
        spatial = rearrange(
            x[:, :, :, dim // 2 :],
            "B C W D -> (B W) C D",
        )
        temporal = rearrange(
            x[:, :, :, : dim // 2],
            "B C W D -> (B C) W D",
        )
        spatial_valid = rearrange(valid, "B C W -> (B W) C")
        temporal_valid = rearrange(valid, "B C W -> (B C) W")

        spatial = self.spatial_attn(spatial, spatial_valid)
        temporal = self.time_attn(temporal, temporal_valid)

        spatial = rearrange(
            spatial,
            "(B W) C D -> B C W D",
            B=batch,
        )
        temporal = rearrange(
            temporal,
            "(B C) W D -> B C W D",
            B=batch,
        )
        return torch.cat([spatial, temporal], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = self._valid_grid(x, sensor_mask, time_mask)
        valid_expanded = valid.unsqueeze(-1).to(dtype=x.dtype)

        x = x * valid_expanded
        x = (
            x
            + self._attn_operator(
                self.pre_attn_norm(x),
                valid,
            )
        ) * valid_expanded
        x = (
            x
            + self.ff(self.pre_ff_norm(x))
        ) * valid_expanded
        return x


class MaskedSpatialTemporalEncoder(nn.Module):
    """Stack with the same state-dict layout as SpatialTemporalEncoder."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList(
            [
                MaskedSpatialTemporalAttentionBlock(
                    n_dim=dim,
                    n_head=heads,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(depth)
            ]
        )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        x: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    layer,
                    x,
                    sensor_mask,
                    time_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    sensor_mask=sensor_mask,
                    time_mask=time_mask,
                )
        return x


class CrissCrossTransformerModule(LegacyCrissCrossTransformerModule):
    """Continuity-aware EEG adaptation of the MEG-XL model."""

    def __init__(self, *args, num_sensor_types: int = 3, **kwargs) -> None:
        # Keep the original two-row MEG coil embedding. This makes
        # sensor_type_layer.weight directly compatible with MEG-XL checkpoints.
        super().__init__(*args, num_sensor_types=2, **kwargs)
        self.requested_num_sensor_types = int(num_sensor_types)

        num_layers = int(self.hparams.num_layers)
        num_heads = int(self.hparams.num_heads)
        self.criss_cross_transformer = MaskedSpatialTemporalEncoder(
            dim=self.latent_dim,
            depth=num_layers,
            heads=num_heads,
            dropout=0.1,
            causal=False,
        )

        # Modality is independent from the MEG coil subtype. Zero initialization
        # leaves a transferred MEG checkpoint behavior unchanged at step zero.
        self.modality_layer = nn.Embedding(2, self.latent_dim)
        nn.init.zeros_(self.modality_layer.weight)

    @staticmethod
    def _encoded_time_mask(
        raw_mask: Optional[torch.Tensor],
        encoded_steps: int,
        batch_size: int,
        raw_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if raw_mask is None:
            return torch.ones(
                batch_size,
                encoded_steps,
                dtype=torch.bool,
                device=device,
            )
        if raw_mask.shape != (batch_size, raw_steps):
            raise ValueError(
                f"Expected raw time mask {(batch_size, raw_steps)}, "
                f"got {tuple(raw_mask.shape)}"
            )
        pooled = F.adaptive_avg_pool1d(
            raw_mask.float().unsqueeze(1),
            encoded_steps,
        ).squeeze(1)
        # A token is valid only when its complete receptive interval is valid.
        return pooled >= (1.0 - 1e-6)

    def _construct_embeddings_with_modalities(
        self,
        codes: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
        sensor_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, quantizers, encoded_steps = codes.shape
        codes = codes.permute(0, 1, 3, 2)

        embedded_levels = []
        for q in range(quantizers):
            codebook = getattr(self, f"biocodec_embedding_{q}")
            embedded_levels.append(
                F.embedding(codes[..., q].long(), codebook)
            )
        embedded_levels = torch.stack(embedded_levels, dim=3)
        embeddings = self.rvq_projector(
            embedded_levels.reshape(
                batch,
                channels,
                encoded_steps,
                quantizers * self.codebook_dim,
            )
        )

        if sensor_mask is None:
            sensor_valid = torch.ones(
                batch,
                channels,
                dtype=torch.bool,
                device=embeddings.device,
            )
        else:
            sensor_valid = sensor_mask.bool()

        is_eeg = sensor_type.long() == EEG_SENSOR_TYPE_ID
        is_meg = ~is_eeg
        modality_ids = torch.where(
            is_eeg,
            torch.full_like(sensor_type.long(), EEG_MODALITY_ID),
            torch.full_like(sensor_type.long(), MEG_MODALITY_ID),
        )

        pos_fourier = self.position_fourier_emb(
            sensor_xyz.reshape(batch * channels, 3)
        ).reshape(batch, channels, -1)
        pos_emb = self.position_projector(pos_fourier)

        ori_fourier = self.orientation_fourier_emb(
            sensor_abc.reshape(batch * channels, 3)
        ).reshape(batch, channels, -1)
        ori_emb = self.orientation_projector(ori_fourier)
        ori_emb = ori_emb * is_meg.unsqueeze(-1).to(ori_emb.dtype)

        coil_types = sensor_type.long().clamp(
            min=GRAD_SENSOR_TYPE_ID,
            max=MAG_SENSOR_TYPE_ID,
        )
        coil_emb = self.sensor_type_layer(coil_types)
        coil_emb = coil_emb * is_meg.unsqueeze(-1).to(coil_emb.dtype)

        modality_emb = self.modality_layer(modality_ids)
        sensor_emb = pos_emb + ori_emb + coil_emb + modality_emb
        sensor_emb = sensor_emb * sensor_valid.unsqueeze(-1).to(
            sensor_emb.dtype
        )

        embeddings = embeddings + sensor_emb.unsqueeze(2)
        embeddings = embeddings * sensor_valid[:, :, None, None].to(
            embeddings.dtype
        )
        return embeddings, codes

    def _generate_target_block_mask(
        self,
        batch_size: int,
        channels: int,
        encoded_steps: int,
        sensor_mask: Optional[torch.Tensor],
        target_time_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        mask = torch.zeros(
            batch_size,
            encoded_steps,
            dtype=torch.bool,
            device=device,
        )
        if self.num_subsegments_to_mask <= 0:
            return mask.unsqueeze(1).expand(-1, channels, -1), 0.0

        block_length = max(1, int(self.mask_length))

        for batch_idx in range(batch_size):
            valid_starts = [
                start
                for start in range(
                    0,
                    max(0, encoded_steps - block_length + 1),
                )
                if bool(
                    target_time_mask[
                        batch_idx,
                        start : start + block_length,
                    ].all()
                )
            ]
            if not valid_starts:
                continue

            order = torch.randperm(
                len(valid_starts),
                device=device,
            )
            occupied = torch.zeros(
                encoded_steps,
                dtype=torch.bool,
                device=device,
            )
            selected = 0
            for selected_idx in order.tolist():
                start = valid_starts[selected_idx]
                end = start + block_length
                if bool(occupied[start:end].any()):
                    continue
                mask[batch_idx, start:end] = True
                occupied[start:end] = True
                selected += 1
                if selected >= self.num_subsegments_to_mask:
                    break

        mask = mask.unsqueeze(1).expand(
            batch_size,
            channels,
            encoded_steps,
        )
        if sensor_mask is not None:
            mask = mask & sensor_mask.bool().unsqueeze(-1)

        if sensor_mask is None:
            denominator = target_time_mask.sum() * channels
        else:
            denominator = (
                target_time_mask.unsqueeze(1)
                & sensor_mask.bool().unsqueeze(-1)
            ).sum()
        ratio = (
            (mask.sum().float() / denominator.float()).item()
            if int(denominator.item()) > 0
            else 0.0
        )
        return mask, ratio

    def forward(
        self,
        raw_meg: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
        collect_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        timing: Dict[str, float] = {}
        if collect_timing:
            start_total = time.perf_counter()

        codes = self._tokenize_multichannel(raw_meg)
        batch, channels, _, encoded_steps = codes.shape
        if collect_timing:
            after_tokenize = time.perf_counter()
            timing["tokenize_ms"] = (
                after_tokenize - start_total
            ) * 1000.0

        token_time_mask = self._encoded_time_mask(
            time_mask,
            encoded_steps,
            batch,
            raw_meg.shape[-1],
            raw_meg.device,
        )
        token_target_mask = self._encoded_time_mask(
            target_mask,
            encoded_steps,
            batch,
            raw_meg.shape[-1],
            raw_meg.device,
        )
        token_target_mask &= token_time_mask

        embeddings, codes_reordered = (
            self._construct_embeddings_with_modalities(
                codes,
                sensor_xyz,
                sensor_abc,
                sensor_type,
                sensor_mask,
            )
        )
        if collect_timing:
            after_embeddings = time.perf_counter()
            timing["embeddings_ms"] = (
                after_embeddings - after_tokenize
            ) * 1000.0

        mask_info: Dict[str, torch.Tensor | float] = {}
        if apply_mask:
            mask, mask_ratio = self._generate_target_block_mask(
                batch,
                channels,
                encoded_steps,
                sensor_mask,
                token_target_mask,
                embeddings.device,
            )
            embeddings = self._apply_mask(
                embeddings,
                codes_reordered,
                mask,
            )
            mask_info["mask"] = mask
            mask_info["mask_ratio"] = mask_ratio

        if collect_timing:
            after_mask = time.perf_counter()
            timing["masking_ms"] = (
                after_mask - after_embeddings
            ) * 1000.0

        transformer_out = self.criss_cross_transformer(
            embeddings,
            sensor_mask=sensor_mask,
            time_mask=token_time_mask,
        )

        if collect_timing:
            after_transformer = time.perf_counter()
            timing["transformer_ms"] = (
                after_transformer - after_mask
            ) * 1000.0

        logits = self.output_head(transformer_out).reshape(
            batch,
            channels,
            encoded_steps,
            self.n_q,
            self.vocab_size,
        )

        if collect_timing:
            after_output = time.perf_counter()
            timing["output_proj_ms"] = (
                after_output - after_transformer
            ) * 1000.0
            timing["total_ms"] = (
                after_output - start_total
            ) * 1000.0

        result = {
            "logits": logits,
            "codes": codes_reordered,
            "features": transformer_out,
            "token_time_mask": token_time_mask,
            "token_target_mask": token_target_mask,
            **mask_info,
        }
        if collect_timing:
            result["timing"] = timing
        return result

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 7:
            return batch
        if len(batch) == 5:
            raw, xyzdir, types, sensor_mask, dataset_ids = batch
            time_mask = torch.ones(
                raw.shape[0],
                raw.shape[-1],
                dtype=torch.bool,
                device=raw.device,
            )
            return (
                raw,
                xyzdir,
                types,
                sensor_mask,
                time_mask,
                time_mask,
                dataset_ids,
            )
        raise ValueError(
            "Expected a 7-item continuity-aware EEG batch or legacy 5-item "
            f"batch, got {len(batch)} items"
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        (
            raw_meg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            time_mask,
            target_mask,
            dataset_ids,
        ) = self._unpack_batch(batch)
        sensor_xyz = sensor_xyzdir[..., :3]
        sensor_abc = sensor_xyzdir[..., 3:]

        start = time.perf_counter()
        output = self.forward(
            raw_meg,
            sensor_xyz,
            sensor_abc,
            sensor_types,
            sensor_mask=sensor_mask,
            time_mask=time_mask,
            target_mask=target_mask,
            apply_mask=True,
            collect_timing=True,
        )
        after_forward = time.perf_counter()

        metrics = self._compute_metrics(
            output["logits"],
            output["codes"],
            output["mask"],
            sensor_mask,
            raw_meg=raw_meg,
            sensor_xyz=sensor_xyz,
            sensor_abc=sensor_abc,
            sensor_types=sensor_types,
            dataset_ids=dataset_ids,
        )
        after_loss = time.perf_counter()

        self.log(
            "train/loss",
            metrics["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/accuracy",
            metrics["accuracy"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for q in range(self.n_q):
            self.log(
                f"train/accuracy_q{q}",
                metrics[f"accuracy_q{q}"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        for key, value in output["timing"].items():
            self.log(
                f"timing/{key}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "timing/loss_calc_ms",
            (after_loss - after_forward) * 1000.0,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "timing/total_step_ms",
            (after_loss - start) * 1000.0,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/mask_ratio",
            output["mask_ratio"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/target_token_ratio",
            output["token_target_mask"].float().mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return metrics["loss"]

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        (
            raw_meg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            time_mask,
            target_mask,
            dataset_ids,
        ) = self._unpack_batch(batch)
        sensor_xyz = sensor_xyzdir[..., :3]
        sensor_abc = sensor_xyzdir[..., 3:]

        output = self.forward(
            raw_meg,
            sensor_xyz,
            sensor_abc,
            sensor_types,
            sensor_mask=sensor_mask,
            time_mask=time_mask,
            target_mask=target_mask,
            apply_mask=True,
        )
        logits = output["logits"].contiguous()
        codes = output["codes"].contiguous()
        mask = output["mask"].contiguous()

        global_metrics = self._compute_metrics(
            logits,
            codes,
            mask,
            sensor_mask,
            raw_meg=raw_meg,
            sensor_xyz=sensor_xyz,
            sensor_abc=sensor_abc,
            sensor_types=sensor_types,
            dataset_ids=dataset_ids,
        )
        self.log(
            "val/loss",
            global_metrics["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/accuracy",
            global_metrics["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for q in range(self.n_q):
            self.log(
                f"val/accuracy_q{q}",
                global_metrics[f"accuracy_q{q}"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "val/target_token_ratio",
            output["token_target_mask"].float().mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        if dataset_ids is not None:
            mapping = {}
            if self.trainer is not None and hasattr(
                self.trainer,
                "datamodule",
            ):
                mapping = (
                    self.trainer.datamodule.get_dataset_name_mapping()
                )
            for dataset_id in torch.unique(dataset_ids):
                indices = torch.where(dataset_ids == dataset_id)[0]
                if indices.numel() == 0:
                    continue
                sub_metrics = self._compute_metrics(
                    torch.index_select(logits, 0, indices),
                    torch.index_select(codes, 0, indices),
                    torch.index_select(mask, 0, indices),
                    torch.index_select(sensor_mask, 0, indices),
                )
                dataset_id_int = int(dataset_id.item())
                name = mapping.get(
                    dataset_id_int,
                    f"dataset_{dataset_id_int}",
                )
                self.log(
                    f"val/{name}_loss",
                    sub_metrics["loss"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"val/{name}_acc",
                    sub_metrics["accuracy"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        return global_metrics["loss"]


__all__ = [
    "CrissCrossTransformerModule",
    "MaskedSelfAttention",
    "MaskedSpatialTemporalEncoder",
]
