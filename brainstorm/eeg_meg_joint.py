"""Optional joint EEG/MEG training utilities for the continuous MEG-XL entrypoint.

The base Criss-Cross Transformer architecture is not modified. These helpers are
used only when ``data.mix_eeg_meg=true``:

* one continuous EEG batch and one MEG batch are processed per optimizer step;
* both modalities share the tokenizer, RVQ projector, transformer and output head;
* EEG can reuse the MEG magnetometer sensor-type embedding while its orientation
  contribution remains disabled by the base model's EEG handling;
* a MEG-XL checkpoint with two sensor-type rows can be expanded safely to the
  three-row EEG/MEG model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from brainstorm.data.eeg_continuous_masked_datamodule import MultiEEGDataModule
from brainstorm.data.multi_datamodule import MultiMEGDataModule
from brainstorm.models.criss_cross_transformer import (
    EEG_SENSOR_TYPE_ID,
    CrissCrossTransformerModule,
)


class JointEEGMEGDataModule(pl.LightningDataModule):
    """Expose paired EEG and MEG loaders without concatenating their channels."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.meg_datamodule: Optional[MultiMEGDataModule] = None
        self.eeg_datamodule: Optional[MultiEEGDataModule] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        if self.meg_datamodule is not None and self.eeg_datamodule is not None:
            return

        data = self.cfg.data
        training = self.cfg.training
        joint = self.cfg.joint
        tokenizer_name = str(self.cfg.model.get("tokenizer_name", "biocodec"))

        self.meg_datamodule = MultiMEGDataModule(
            datasets_config=OmegaConf.to_container(
                self.cfg.meg_datasets_config,
                resolve=True,
            ),
            segment_length=float(data.segment_length),
            cache_dir=str(data.meg_cache_dir),
            l_freq=float(data.l_freq),
            h_freq=float(data.h_freq),
            target_sfreq=float(data.target_sfreq),
            batch_size=int(joint.get("meg_batch_size", training.batch_size)),
            num_workers=int(training.num_workers),
            pin_memory=bool(training.pin_memory),
            persistent_workers=bool(training.persistent_workers),
            use_recording_sampler=bool(training.use_recording_sampler),
            sampler_seed=int(training.sampler_seed),
            debug_mode=bool(data.get("debug_mode", False)),
            shuffle_segments=bool(data.get("shuffle_segments", False)),
            shuffle_segment_duration=float(
                data.get("shuffle_segment_duration", 3.0)
            ),
            recording_subsample_prop=data.get(
                "meg_recording_subsample_prop",
                None,
            ),
        )

        self.eeg_datamodule = MultiEEGDataModule(
            datasets_config=OmegaConf.to_container(
                self.cfg.datasets_config,
                resolve=True,
            ),
            segment_length=float(data.segment_length),
            subsegment_duration=float(data.get("subsegment_duration", 3.0)),
            words_per_segment=int(data.get("words_per_segment", 50)),
            window_onset_offset=float(data.get("window_onset_offset", -0.5)),
            cache_dir=str(data.cache_dir),
            l_freq=float(data.l_freq),
            h_freq=float(data.h_freq),
            target_sfreq=float(data.target_sfreq),
            batch_size=int(joint.get("eeg_batch_size", training.batch_size)),
            num_workers=int(training.num_workers),
            pin_memory=bool(training.pin_memory),
            persistent_workers=bool(training.persistent_workers),
            use_recording_sampler=bool(training.use_recording_sampler),
            sampler_seed=int(training.sampler_seed),
            debug_mode=bool(data.get("debug_mode", False)),
            max_channel_dim=data.get("max_channel_dim", None),
            infer_max_channel_dim=bool(data.get("infer_max_channel_dim", True)),
            recording_subsample_prop=data.get(
                "recording_subsample_prop",
                None,
            ),
            allow_missing_word_alignment=bool(
                data.get("allow_missing_word_alignment", False)
            ),
            tokenizer_name=tokenizer_name,
        )

        self.meg_datamodule.setup("fit")
        self.eeg_datamodule.setup("fit")

    def _require_modules(self) -> Tuple[MultiMEGDataModule, MultiEEGDataModule]:
        if self.meg_datamodule is None or self.eeg_datamodule is None:
            raise RuntimeError("Call JointEEGMEGDataModule.setup('fit') first.")
        return self.meg_datamodule, self.eeg_datamodule

    def train_dataloader(self):
        meg, eeg = self._require_modules()
        return CombinedLoader(
            {
                "meg": meg.train_dataloader(),
                "eeg": eeg.train_dataloader(),
            },
            mode=str(self.cfg.joint.get("train_loader_mode", "max_size_cycle")),
        )

    def val_dataloader(self):
        meg, eeg = self._require_modules()
        return CombinedLoader(
            {
                "meg": meg.val_dataloader(),
                "eeg": eeg.val_dataloader(),
            },
            mode=str(self.cfg.joint.get("val_loader_mode", "min_size")),
        )

    def num_training_batches(self) -> int:
        """Return the CombinedLoader length without relying on its iterator state."""
        meg, eeg = self._require_modules()
        meg_batches = len(meg.train_dataloader())
        eeg_batches = len(eeg.train_dataloader())
        mode = str(self.cfg.joint.get("train_loader_mode", "max_size_cycle"))
        if mode in {"max_size", "max_size_cycle"}:
            return max(meg_batches, eeg_batches)
        if mode == "min_size":
            return min(meg_batches, eeg_batches)
        if mode == "sequential":
            return meg_batches + eeg_batches
        raise ValueError(f"Unsupported CombinedLoader mode: {mode}")

    def teardown(self, stage: Optional[str] = None) -> None:
        if self.meg_datamodule is not None:
            self.meg_datamodule.teardown(stage)
        if self.eeg_datamodule is not None:
            self.eeg_datamodule.teardown(stage)


class SharedEEGMEGCrissCrossTransformerModule(CrissCrossTransformerModule):
    """Unchanged MEG-XL backbone trained on one EEG and one MEG batch per step."""

    def __init__(
        self,
        *args: Any,
        meg_loss_weight: float = 1.0,
        eeg_loss_weight: float = 1.0,
        eeg_as_meg_sensor_type: bool = True,
        eeg_meg_sensor_type_id: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if meg_loss_weight < 0 or eeg_loss_weight < 0:
            raise ValueError("Modality loss weights must be non-negative.")
        if meg_loss_weight + eeg_loss_weight <= 0:
            raise ValueError("At least one modality loss weight must be positive.")

        self.meg_loss_weight = float(meg_loss_weight)
        self.eeg_loss_weight = float(eeg_loss_weight)
        self.eeg_as_meg_sensor_type = bool(eeg_as_meg_sensor_type)
        self.eeg_meg_sensor_type_id = int(eeg_meg_sensor_type_id)

        if not 0 <= self.eeg_meg_sensor_type_id < self.num_sensor_types:
            raise ValueError(
                "eeg_meg_sensor_type_id must be inside the sensor-type vocabulary; "
                f"got {self.eeg_meg_sensor_type_id} for {self.num_sensor_types} types."
            )

    def _construct_embeddings(
        self,
        codes: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, reordered_codes = super()._construct_embeddings(
            codes,
            sensor_xyz,
            sensor_abc,
            sensor_type,
        )

        if self.eeg_as_meg_sensor_type:
            original_type_embedding = self.sensor_type_layer(sensor_type.long())
            remapped_type = sensor_type.long().clone()
            remapped_type[remapped_type == EEG_SENSOR_TYPE_ID] = (
                self.eeg_meg_sensor_type_id
            )
            shared_type_embedding = self.sensor_type_layer(remapped_type)
            embeddings = embeddings + (
                shared_type_embedding - original_type_embedding
            ).unsqueeze(2)

        return embeddings, reordered_codes

    @staticmethod
    def _feature_centroid(
        features: torch.Tensor,
        sensor_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sensor_mask is None:
            return features.mean(dim=(0, 1, 2))

        weights = sensor_mask.to(features.dtype).unsqueeze(-1).unsqueeze(-1)
        numerator = (features * weights).sum(dim=(0, 1, 2))
        denominator = weights.sum(dim=(0, 1, 2)).clamp_min(1.0)
        return numerator / denominator

    def _run_modality_batch(
        self,
        batch,
        collect_timing: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        (
            raw_signal,
            sensor_xyzdir,
            sensor_types,
            sensor_mask,
            target_mask,
            dataset_ids,
        ) = self._unpack_batch(batch)

        sensor_xyz = sensor_xyzdir[..., :3]
        sensor_abc = sensor_xyzdir[..., 3:]
        output = self.forward(
            raw_signal,
            sensor_xyz,
            sensor_abc,
            sensor_types,
            sensor_mask=sensor_mask,
            target_mask=target_mask,
            apply_mask=True,
            collect_timing=collect_timing,
        )
        metrics = self._compute_metrics(
            output["logits"],
            output["codes"],
            output["mask"],
            sensor_mask,
            raw_meg=raw_signal,
            sensor_xyz=sensor_xyz,
            sensor_abc=sensor_abc,
            sensor_types=sensor_types,
            dataset_ids=dataset_ids,
        )
        centroid = self._feature_centroid(output["features"], sensor_mask)
        return output, metrics, centroid

    def _log_modality_metrics(
        self,
        stage: str,
        modality: str,
        metrics: Dict[str, torch.Tensor],
        on_step: bool,
    ) -> None:
        self.log(
            f"{stage}/{modality}_loss",
            metrics["loss"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
            batch_size=1,
        )
        self.log(
            f"{stage}/{modality}_accuracy",
            metrics["accuracy"],
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        if not isinstance(batch, dict) or "meg" not in batch or "eeg" not in batch:
            raise ValueError(
                "Joint training expects a CombinedLoader batch with 'meg' and 'eeg'."
            )

        collect_timing = stage == "train"
        meg_output, meg_metrics, meg_centroid = self._run_modality_batch(
            batch["meg"],
            collect_timing=collect_timing,
        )
        eeg_output, eeg_metrics, eeg_centroid = self._run_modality_batch(
            batch["eeg"],
            collect_timing=collect_timing,
        )

        weight_sum = self.meg_loss_weight + self.eeg_loss_weight
        loss = (
            self.meg_loss_weight * meg_metrics["loss"]
            + self.eeg_loss_weight * eeg_metrics["loss"]
        ) / weight_sum

        cosine = F.cosine_similarity(
            meg_centroid.unsqueeze(0),
            eeg_centroid.unsqueeze(0),
            dim=-1,
        ).squeeze(0)
        centroid_distance = torch.linalg.vector_norm(meg_centroid - eeg_centroid)
        meg_norm = torch.linalg.vector_norm(meg_centroid)
        eeg_norm = torch.linalg.vector_norm(eeg_centroid)
        summed_norm = torch.linalg.vector_norm(meg_centroid + eeg_centroid)
        additivity_ratio = summed_norm / (meg_norm + eeg_norm).clamp_min(1e-8)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=1,
        )
        self._log_modality_metrics(
            stage,
            "meg",
            meg_metrics,
            on_step=(stage == "train"),
        )
        self._log_modality_metrics(
            stage,
            "eeg",
            eeg_metrics,
            on_step=(stage == "train"),
        )

        cross_metrics = {
            "centroid_cosine": cosine,
            "centroid_l2": centroid_distance,
            "meg_centroid_norm": meg_norm,
            "eeg_centroid_norm": eeg_norm,
            "summed_centroid_norm": summed_norm,
            "additivity_ratio": additivity_ratio,
        }
        for name, value in cross_metrics.items():
            self.log(
                f"{stage}/cross_modal_{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )

        if collect_timing:
            for modality, output in (("meg", meg_output), ("eeg", eeg_output)):
                for name, value in output.get("timing", {}).items():
                    self.log(
                        f"timing/{modality}_{name}",
                        value,
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=1,
                    )

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._shared_step(batch, stage="val")


def load_megxl_checkpoint_for_eeg(
    model: torch.nn.Module,
    checkpoint_path: str | None,
    *,
    copy_meg_type_to_eeg: bool = True,
    meg_sensor_type_id: int = 1,
    eeg_sensor_type_id: int = EEG_SENSOR_TYPE_ID,
) -> Dict[str, Any]:
    """Load MEG-XL and expand its sensor-type table for the EEG row.

    All shape-compatible tensors are loaded normally. The only supported shape
    adaptation is ``sensor_type_layer.weight`` from two MEG rows to the larger
    EEG/MEG table. The EEG row is initialized from the selected MEG row.
    """

    if not checkpoint_path:
        return {"requested": False, "loaded": False, "reason": "no checkpoint path"}

    path = Path(checkpoint_path)
    if not path.exists():
        return {
            "requested": True,
            "loaded": False,
            "reason": f"checkpoint not found: {path}",
        }

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()

    compatible: Dict[str, torch.Tensor] = {}
    skipped_shape = []
    skipped_missing = []
    adapted = []

    for key, value in state_dict.items():
        if key not in current:
            skipped_missing.append(key)
            continue

        target = current[key]
        if tuple(target.shape) == tuple(value.shape):
            compatible[key] = value
            continue

        can_expand_sensor_types = (
            key.endswith("sensor_type_layer.weight")
            and value.ndim == 2
            and target.ndim == 2
            and value.shape[1] == target.shape[1]
            and value.shape[0] <= target.shape[0]
        )
        if can_expand_sensor_types:
            expanded = target.detach().clone()
            expanded[: value.shape[0]] = value
            if copy_meg_type_to_eeg:
                if not 0 <= meg_sensor_type_id < value.shape[0]:
                    raise ValueError(
                        f"MEG sensor type {meg_sensor_type_id} is unavailable in "
                        f"checkpoint tensor with {value.shape[0]} rows."
                    )
                if not 0 <= eeg_sensor_type_id < expanded.shape[0]:
                    raise ValueError(
                        f"EEG sensor type {eeg_sensor_type_id} is unavailable in "
                        f"model tensor with {expanded.shape[0]} rows."
                    )
                expanded[eeg_sensor_type_id] = value[meg_sensor_type_id]
            compatible[key] = expanded
            adapted.append(
                {
                    "key": key,
                    "checkpoint": list(value.shape),
                    "model": list(target.shape),
                    "copied_row": [meg_sensor_type_id, eeg_sensor_type_id],
                }
            )
            continue

        skipped_shape.append(
            {
                "key": key,
                "checkpoint": list(value.shape),
                "model": list(target.shape),
            }
        )

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "requested": True,
        "loaded": True,
        "checkpoint": str(path),
        "loaded_keys": len(compatible),
        "total_checkpoint_keys": len(state_dict),
        "adapted_sensor_type_tensors": adapted,
        "missing_after_load": list(missing),
        "unexpected_after_load": list(unexpected),
        "skipped_missing_count": len(skipped_missing),
        "skipped_shape_count": len(skipped_shape),
        "skipped_shape_examples": skipped_shape[:20],
    }
