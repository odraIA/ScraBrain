import time
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from brainstorm.models.attentional.attn import SpatialTemporalEncoder
from brainstorm.models.spatial_attention import GaussianFourierEmb3D


EEG_SENSOR_TYPE_ID = 2


class CrissCrossTransformerModule(pl.LightningModule):
    """MEG-XL criss-cross transformer for multi-channel MEG/EEG signals.

    EEG uses the same tokenizer, embeddings, criss-cross encoder, output head,
    loss, and sensor-mask semantics as MEG-XL. The only modality-specific
    adaptations are:

    - orientation embeddings are disabled for sensors whose type is EEG;
    - an optional target-time mask can restrict where temporal reconstruction
      blocks are sampled (used for listening intervals inside listeningcovert).
    """

    strict_loading = False

    def __init__(
        self,
        tokenizer: nn.Module,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        vocab_size: int = 256,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        training_steps: int = 10000,
        mask_duration: float = 3.0,
        num_subsegments_to_mask: int = 1,
        sampling_rate: int = 250,
        fourier_pos_dim: int = 250,
        num_sensor_types: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        if num_subsegments_to_mask < 0:
            raise ValueError(
                "num_subsegments_to_mask must be non-negative, got "
                f"{num_subsegments_to_mask}"
            )

        self.tokenizer = tokenizer
        for parameter in self.tokenizer.parameters():
            parameter.requires_grad = False
        self.tokenizer.eval()

        legacy_quantizer = getattr(tokenizer, "quantizer", None)
        self.n_q = int(
            getattr(tokenizer, "n_q", getattr(legacy_quantizer, "n_q", 0))
        )
        tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size", vocab_size))
        if int(vocab_size) != tokenizer_vocab_size:
            print(
                f"Tokenizer vocab_size ({tokenizer_vocab_size}) differs from "
                f"checkpoint/model vocab_size ({vocab_size}); using tokenizer value."
            )

        self.vocab_size = tokenizer_vocab_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.mask_duration = mask_duration
        self.num_subsegments_to_mask = num_subsegments_to_mask
        self.sampling_rate = sampling_rate
        self.num_sensor_types = num_sensor_types

        self.tokenizer_downsample_ratio = int(
            getattr(tokenizer, "downsample_ratio", 12)
        )
        self.biocodec_downsample_ratio = self.tokenizer_downsample_ratio
        mask_samples = round(mask_duration * sampling_rate)
        self.mask_length = mask_samples // self.tokenizer_downsample_ratio

        tokenizer_embeddings = []
        for quantizer_idx in range(self.n_q):
            if hasattr(tokenizer, "codebook_embedding"):
                codebook_embedding = tokenizer.codebook_embedding(quantizer_idx)
            else:
                codebook_embedding = (
                    tokenizer.quantizer.vq.layers[quantizer_idx]._codebook.embed
                )
            self.register_buffer(
                f"biocodec_embedding_{quantizer_idx}",
                codebook_embedding,
            )
            tokenizer_embeddings.append(codebook_embedding)

        self.codebook_dim = tokenizer_embeddings[0].shape[1]
        self.rvq_projector = nn.Linear(
            self.n_q * self.codebook_dim,
            latent_dim,
        )

        self.position_fourier_emb = GaussianFourierEmb3D(
            embed_dim=fourier_pos_dim,
            scale=1.8,
        )
        self.position_projector = nn.Linear(fourier_pos_dim, latent_dim)

        self.orientation_fourier_emb = GaussianFourierEmb3D(
            embed_dim=fourier_pos_dim,
            scale=1.0,
        )
        self.orientation_projector = nn.Linear(fourier_pos_dim, latent_dim)

        self.sensor_type_layer = nn.Embedding(num_sensor_types, latent_dim)
        self.mask_token = nn.Parameter(torch.randn(latent_dim))

        # Original MEG-XL criss-cross attention is kept unchanged.
        self.criss_cross_transformer = SpatialTemporalEncoder(
            dim=latent_dim,
            depth=num_layers,
            heads=num_heads,
            dropout=0.1,
            causal=False,
        )
        self.output_head = nn.Linear(
            latent_dim,
            self.n_q * self.vocab_size,
        )
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        if hasattr(
            self.criss_cross_transformer,
            "gradient_checkpointing_enable",
        ):
            self.criss_cross_transformer.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def _tokenize_multichannel(self, raw_meg: torch.Tensor) -> torch.Tensor:
        batch, channels, samples = raw_meg.shape
        raw_batched = raw_meg.reshape(batch * channels, 1, samples)

        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(raw_batched)
            codes = torch.stack(
                [frame[0] for frame in encoded_frames],
                dim=0,
            )[0]

        _, quantizers, encoded_steps = codes.shape
        return codes.reshape(
            batch,
            channels,
            quantizers,
            encoded_steps,
        )

    def _construct_embeddings(
        self,
        codes: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, quantizers, encoded_steps = codes.shape
        codes = codes.permute(0, 1, 3, 2)

        embedded_levels = []
        for quantizer_idx in range(quantizers):
            codebook = getattr(
                self,
                f"biocodec_embedding_{quantizer_idx}",
            )
            embedded_levels.append(
                F.embedding(codes[..., quantizer_idx].long(), codebook)
            )

        embedded_levels = torch.stack(embedded_levels, dim=3)
        embedded_concat = embedded_levels.reshape(
            batch,
            channels,
            encoded_steps,
            quantizers * self.codebook_dim,
        )
        embeddings = self.rvq_projector(embedded_concat)

        pos_fourier = self.position_fourier_emb(
            sensor_xyz.reshape(batch * channels, 3)
        ).reshape(batch, channels, -1)
        pos_emb = self.position_projector(pos_fourier)
        embeddings = embeddings + pos_emb.unsqueeze(2)

        ori_fourier = self.orientation_fourier_emb(
            sensor_abc.reshape(batch * channels, 3)
        ).reshape(batch, channels, -1)
        ori_emb = self.orientation_projector(ori_fourier)

        # MEG keeps the exact original orientation embedding. EEG electrodes do
        # not represent oriented coils, so their contribution is exactly zero.
        is_meg = sensor_type.long() != EEG_SENSOR_TYPE_ID
        ori_emb = ori_emb * is_meg.unsqueeze(-1).to(ori_emb.dtype)
        embeddings = embeddings + ori_emb.unsqueeze(2)

        type_emb = self.sensor_type_layer(sensor_type.long())
        embeddings = embeddings + type_emb.unsqueeze(2)
        return embeddings, codes

    @staticmethod
    def _downsample_target_mask(
        target_mask: Optional[torch.Tensor],
        encoded_steps: int,
        batch_size: int,
        raw_steps: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Map a raw-sample mask to tokenizer time without changing tokens.

        A token is eligible only when its complete pooled interval belongs to a
        target region. MEG batches pass ``None`` and retain their original mask
        sampling path exactly.
        """
        if target_mask is None:
            return None
        if tuple(target_mask.shape) != (batch_size, raw_steps):
            raise ValueError(
                "Expected target_mask with shape "
                f"{(batch_size, raw_steps)}, got {tuple(target_mask.shape)}"
            )

        pooled = F.adaptive_avg_pool1d(
            target_mask.to(device=device, dtype=torch.float32).unsqueeze(1),
            encoded_steps,
        ).squeeze(1)
        return pooled >= (1.0 - 1e-6)

    def _generate_targeted_temporal_block_mask(
        self,
        batch_size: int,
        n_channels: int,
        n_timesteps: int,
        sensor_mask: Optional[torch.Tensor],
        target_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """Sample non-overlapping MEG-XL blocks inside valid target ranges."""
        temporal_mask = torch.zeros(
            batch_size,
            n_timesteps,
            dtype=torch.bool,
            device=device,
        )

        if self.num_subsegments_to_mask > 0 and self.mask_length > 0:
            if tuple(target_mask.shape) != (batch_size, n_timesteps):
                raise ValueError(
                    "Encoded target_mask must have shape "
                    f"{(batch_size, n_timesteps)}, got "
                    f"{tuple(target_mask.shape)}"
                )

            if n_timesteps >= self.mask_length:
                valid_starts = target_mask.unfold(
                    dimension=1,
                    size=self.mask_length,
                    step=1,
                ).all(dim=-1)

                for batch_idx in range(batch_size):
                    starts = torch.nonzero(
                        valid_starts[batch_idx],
                        as_tuple=False,
                    ).flatten()
                    if starts.numel() == 0:
                        continue

                    order = starts[
                        torch.randperm(starts.numel(), device=device)
                    ]
                    selected = 0
                    for start_tensor in order:
                        start = int(start_tensor.item())
                        end = start + self.mask_length
                        if temporal_mask[batch_idx, start:end].any():
                            continue
                        temporal_mask[batch_idx, start:end] = True
                        selected += 1
                        if selected >= self.num_subsegments_to_mask:
                            break

        mask = temporal_mask.unsqueeze(1).expand(
            batch_size,
            n_channels,
            n_timesteps,
        )
        if sensor_mask is not None:
            mask = mask & sensor_mask.bool().unsqueeze(-1)

        if sensor_mask is not None:
            total_valid_tokens = sensor_mask.sum() * n_timesteps
            if total_valid_tokens > 0:
                mask_ratio = (
                    mask.sum().float() / total_valid_tokens.float()
                ).item()
            else:
                mask_ratio = 0.0
        else:
            mask_ratio = mask.float().mean().item()

        return mask, mask_ratio

    def _generate_temporal_block_mask(
        self,
        B: int,
        n_channels: int,
        n_timesteps: int,
        sensor_mask: Optional[torch.Tensor],
        device: torch.device,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Generate the original MEG-XL mask, optionally target-restricted."""
        if target_mask is not None:
            return self._generate_targeted_temporal_block_mask(
                batch_size=B,
                n_channels=n_channels,
                n_timesteps=n_timesteps,
                sensor_mask=sensor_mask,
                target_mask=target_mask,
                device=device,
            )

        # Original MEG-XL path. Keep this branch unchanged for MEG batches.
        segment_length_seconds = (
            n_timesteps * self.tokenizer_downsample_ratio
        ) / self.sampling_rate
        num_subsegments = int(segment_length_seconds / 3.0)
        if num_subsegments <= 0:
            raise ValueError(
                "Encoded sequence is shorter than one 3-second subsegment"
            )
        subseg_length = n_timesteps // num_subsegments

        if self.num_subsegments_to_mask <= 0:
            mask = torch.zeros(
                B,
                n_channels,
                n_timesteps,
                dtype=torch.bool,
                device=device,
            )
            return mask, 0.0

        actual_n_to_mask = min(
            self.num_subsegments_to_mask,
            num_subsegments,
        )
        if (
            self.num_subsegments_to_mask > num_subsegments
            and not hasattr(self, "_warned_subseg_clamp")
        ):
            print(
                f"Warning: num_subsegments_to_mask "
                f"({self.num_subsegments_to_mask}) exceeds available "
                f"subsegments ({num_subsegments}). Masking all "
                f"{num_subsegments} subsegments instead."
            )
            self._warned_subseg_clamp = True

        probabilities = torch.ones(B, num_subsegments, device=device)
        probabilities = probabilities / num_subsegments
        subseg_indices = torch.multinomial(
            probabilities,
            num_samples=actual_n_to_mask,
            replacement=False,
        ).unsqueeze(-1)

        subseg_starts = subseg_indices * subseg_length
        max_offset = max(0, subseg_length - self.mask_length)
        random_offsets = torch.randint(
            0,
            max_offset + 1,
            (B, actual_n_to_mask, 1),
            device=device,
        )
        start_indices = subseg_starts + random_offsets
        time_indices = torch.arange(
            n_timesteps,
            device=device,
        ).view(1, 1, -1)
        mask_per_subsegment = (
            (time_indices >= start_indices)
            & (time_indices < start_indices + self.mask_length)
        )
        temporal_mask = mask_per_subsegment.any(dim=1)
        mask = temporal_mask.unsqueeze(1).expand(
            B,
            n_channels,
            n_timesteps,
        )

        if sensor_mask is not None:
            mask = mask & sensor_mask.bool().unsqueeze(-1)
            total_valid_tokens = sensor_mask.sum() * n_timesteps
            if total_valid_tokens > 0:
                mask_ratio = (
                    mask.sum().float() / total_valid_tokens.float()
                ).item()
            else:
                mask_ratio = 0.0
        else:
            mask_ratio = mask.float().mean().item()

        return mask, mask_ratio

    def _apply_mask(
        self,
        embeddings: torch.Tensor,
        codes: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, encoded_steps, quantizers = codes.shape
        embedded_levels = []
        for quantizer_idx in range(quantizers):
            codebook = getattr(
                self,
                f"biocodec_embedding_{quantizer_idx}",
            )
            embedded_levels.append(
                F.embedding(codes[..., quantizer_idx].long(), codebook)
            )

        embedded_levels = torch.stack(embedded_levels, dim=3)
        embedded_concat = embedded_levels.reshape(
            batch,
            channels,
            encoded_steps,
            quantizers * self.codebook_dim,
        )
        original_embeddings = self.rvq_projector(embedded_concat)

        return torch.where(
            mask.unsqueeze(-1),
            embeddings - original_embeddings + self.mask_token,
            embeddings,
        )

    def forward(
        self,
        raw_meg: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
        collect_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        timing: Dict[str, float] = {}
        if collect_timing:
            start_time = time.perf_counter()

        codes = self._tokenize_multichannel(raw_meg)
        batch, channels, _, encoded_steps = codes.shape
        if collect_timing:
            tokenized_time = time.perf_counter()
            timing["tokenize_ms"] = (
                tokenized_time - start_time
            ) * 1000

        embeddings, reordered_codes = self._construct_embeddings(
            codes,
            sensor_xyz,
            sensor_abc,
            sensor_type,
        )
        if collect_timing:
            embedded_time = time.perf_counter()
            timing["embeddings_ms"] = (
                embedded_time - tokenized_time
            ) * 1000

        encoded_target_mask = self._downsample_target_mask(
            target_mask=target_mask,
            encoded_steps=encoded_steps,
            batch_size=batch,
            raw_steps=raw_meg.shape[-1],
            device=embeddings.device,
        )

        mask_info: Dict[str, Any] = {}
        if apply_mask:
            mask, mask_ratio = self._generate_temporal_block_mask(
                batch,
                channels,
                encoded_steps,
                sensor_mask,
                embeddings.device,
                target_mask=encoded_target_mask,
            )
            embeddings = self._apply_mask(
                embeddings,
                reordered_codes,
                mask,
            )
            mask_info["mask"] = mask
            mask_info["mask_ratio"] = mask_ratio

        if collect_timing:
            masked_time = time.perf_counter()
            timing["masking_ms"] = (
                masked_time - embedded_time
            ) * 1000

        # No sensor/time mask is passed here: this is the original MEG-XL
        # attention behavior, including its existing padded-sensor semantics.
        transformer_out = self.criss_cross_transformer(embeddings)
        if collect_timing:
            transformed_time = time.perf_counter()
            timing["transformer_ms"] = (
                transformed_time - masked_time
            ) * 1000

        logits = self.output_head(transformer_out).reshape(
            batch,
            channels,
            encoded_steps,
            self.n_q,
            self.vocab_size,
        )
        if collect_timing:
            output_time = time.perf_counter()
            timing["output_proj_ms"] = (
                output_time - transformed_time
            ) * 1000
            timing["total_ms"] = (
                output_time - start_time
            ) * 1000

        result = {
            "logits": logits,
            "codes": reordered_codes,
            "features": transformer_out,
            **mask_info,
        }
        if collect_timing:
            result["timing"] = timing
        return result

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        codes: torch.Tensor,
        mask: torch.Tensor,
        sensor_mask: Optional[torch.Tensor],
        raw_meg: Optional[torch.Tensor] = None,
        sensor_xyz: Optional[torch.Tensor] = None,
        sensor_abc: Optional[torch.Tensor] = None,
        sensor_types: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del raw_meg, sensor_xyz, sensor_abc, sensor_types, dataset_ids

        _, _, encoded_steps, _, vocabulary_size = logits.shape
        if sensor_mask is not None:
            valid_mask = mask & sensor_mask.bool().unsqueeze(-1).expand(
                -1,
                -1,
                encoded_steps,
            )
        else:
            valid_mask = mask

        logits_flat = logits.reshape(-1, self.n_q, vocabulary_size)
        codes_flat = codes.reshape(-1, self.n_q)
        valid_flat = valid_mask.reshape(-1)
        logits_masked = logits_flat[valid_flat]
        codes_masked = codes_flat[valid_flat]

        metrics: Dict[str, torch.Tensor] = {}
        if logits_masked.numel() == 0:
            metrics["loss"] = torch.tensor(
                0.0,
                device=logits.device,
                requires_grad=True,
            )
            metrics["accuracy"] = torch.tensor(0.0, device=logits.device)
            for quantizer_idx in range(self.n_q):
                metrics[f"accuracy_q{quantizer_idx}"] = torch.tensor(
                    0.0,
                    device=logits.device,
                )
            return metrics

        loss = torch.tensor(0.0, device=logits.device)
        accuracies = []
        for quantizer_idx in range(self.n_q):
            logits_q = logits_masked[:, quantizer_idx, :]
            targets_q = codes_masked[:, quantizer_idx]
            loss = loss + F.cross_entropy(logits_q, targets_q)
            accuracy_q = (
                logits_q.argmax(dim=-1) == targets_q
            ).float().mean()
            accuracies.append(accuracy_q)
            metrics[f"accuracy_q{quantizer_idx}"] = accuracy_q

        metrics["loss"] = loss / self.n_q
        metrics["accuracy"] = torch.stack(accuracies).mean()
        return metrics

    @staticmethod
    def _unpack_batch(batch):
        """Support original MEG batches and EEG batches with target_mask."""
        if len(batch) == 6:
            (
                raw_meg,
                sensor_xyzdir,
                sensor_types,
                sensor_mask,
                target_mask,
                dataset_ids,
            ) = batch
        elif len(batch) == 5:
            (
                raw_meg,
                sensor_xyzdir,
                sensor_types,
                sensor_mask,
                dataset_ids,
            ) = batch
            target_mask = None
        elif len(batch) == 4:
            (
                raw_meg,
                sensor_xyzdir,
                sensor_types,
                sensor_mask,
            ) = batch
            target_mask = None
            dataset_ids = None
        else:
            raise ValueError(
                "Expected a 4/5-item MEG batch or a 6-item continuous EEG batch"
            )

        return (
            raw_meg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            target_mask,
            dataset_ids,
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        (
            raw_meg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            target_mask,
            dataset_ids,
        ) = self._unpack_batch(batch)

        sensor_xyz = sensor_xyzdir[..., :3]
        sensor_abc = sensor_xyzdir[..., 3:]
        step_start = time.perf_counter()

        output = self.forward(
            raw_meg,
            sensor_xyz,
            sensor_abc,
            sensor_types,
            sensor_mask=sensor_mask,
            target_mask=target_mask,
            apply_mask=True,
            collect_timing=True,
        )
        forward_end = time.perf_counter()

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
        loss_end = time.perf_counter()

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
        for quantizer_idx in range(self.n_q):
            self.log(
                f"train/accuracy_q{quantizer_idx}",
                metrics[f"accuracy_q{quantizer_idx}"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        for name, value in output["timing"].items():
            self.log(
                f"timing/{name}",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "timing/loss_calc_ms",
            (loss_end - forward_end) * 1000,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "timing/total_step_ms",
            (loss_end - step_start) * 1000,
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
        return metrics["loss"]

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        (
            raw_meg,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
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
        for quantizer_idx in range(self.n_q):
            self.log(
                f"val/accuracy_q{quantizer_idx}",
                global_metrics[f"accuracy_q{quantizer_idx}"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if dataset_ids is not None:
            dataset_name_mapping = {}
            if self.trainer is not None and hasattr(
                self.trainer,
                "datamodule",
            ):
                dataset_name_mapping = (
                    self.trainer.datamodule.get_dataset_name_mapping()
                )

            for dataset_id in torch.unique(dataset_ids):
                batch_indices = torch.where(dataset_ids == dataset_id)[0]
                if batch_indices.numel() == 0:
                    continue
                if batch_indices.numel() == 1:
                    batch_indices = batch_indices.repeat(2)

                subset_metrics = self._compute_metrics(
                    torch.index_select(logits, 0, batch_indices),
                    torch.index_select(codes, 0, batch_indices),
                    torch.index_select(mask, 0, batch_indices),
                    torch.index_select(
                        sensor_mask,
                        0,
                        batch_indices,
                    )
                    if sensor_mask is not None
                    else None,
                )
                dataset_id_int = int(dataset_id.item())
                dataset_name = dataset_name_mapping.get(
                    dataset_id_int,
                    f"dataset_{dataset_id_int}",
                )
                self.log(
                    f"val/{dataset_name}_loss",
                    subset_metrics["loss"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"val/{dataset_name}_acc",
                    subset_metrics["accuracy"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return global_metrics["loss"]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "state_dict" not in checkpoint:
            return

        skipped_rope_keys = []
        filtered_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            if "rope_embedding_layer.rotate" in key:
                skipped_rope_keys.append(key)
            else:
                filtered_state_dict[key] = value

        if skipped_rope_keys:
            print("\n" + "=" * 60)
            print("Checkpoint Loading: Skipping RoPE rotation buffers")
            print("=" * 60)
            print(
                f"  Skipped {len(skipped_rope_keys)} RoPE keys "
                "(will auto-expand on first forward pass)"
            )
            print(f"  Example keys: {skipped_rope_keys[:2]}")
            print("=" * 60 + "\n")

        checkpoint["state_dict"] = filtered_state_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_steps,
            eta_min=self.learning_rate * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
