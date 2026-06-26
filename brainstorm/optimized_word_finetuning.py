"""Optimized downstream word-decoding fine-tuning for MEG-XL/EEG-XL.

The implementation preserves the downstream objective, word batch, validation
criterion, and final test protocol while removing work that is unnecessary for
word decoding:

* bfloat16 autocast on supported CUDA devices;
* feature-only CrissCross forward, skipping the pre-training output head;
* one batched MLP call for all word windows instead of one call per word;
* optional gradient checkpointing, disabled by default for this low-memory task;
* validation every epoch but test only once, after loading the best checkpoint;
* no optimizer state for the unused pre-training output head.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from brainstorm import evaluate_criss_cross_word_classification as base
from brainstorm.eval_metrics_history import append_epoch_metrics_history, resolve_checkpoint_dir
from brainstorm.losses.contrastive import SigLipLoss
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule


logger = logging.getLogger(__name__)


def _cuda_autocast_context(device: str, enabled: bool, dtype_name: str):
    device_type = torch.device(device).type
    if not enabled or device_type != "cuda" or not torch.cuda.is_available():
        return nullcontext()

    normalized = str(dtype_name).strip().lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            logger.warning("CUDA device does not support bfloat16; using float32")
            return nullcontext()
        dtype = torch.bfloat16
    elif normalized in {"fp16", "float16", "half"}:
        dtype = torch.float16
    else:
        raise ValueError(
            f"Unsupported training.amp_dtype={dtype_name!r}; expected bfloat16 or float16"
        )
    return torch.autocast(device_type="cuda", dtype=dtype)


def _feature_only_forward(
    model: CrissCrossTransformerModule,
    meg: torch.Tensor,
    sensor_xyz: torch.Tensor,
    sensor_abc: torch.Tensor,
    sensor_types: torch.Tensor,
) -> torch.Tensor:
    """Return CrissCross features without constructing pre-training logits.

    Word decoding never consumes ``output_head`` logits. Calling the public model
    forward would nevertheless project every channel/time token into all neural
    tokenizer codebooks. This reproduces its unmasked feature path exactly and
    stops immediately after the criss-cross encoder.
    """

    codes = model._tokenize_multichannel(meg)
    embeddings, _reordered_codes = model._construct_embeddings(
        codes,
        sensor_xyz,
        sensor_abc,
        sensor_types,
    )
    return model.criss_cross_transformer(embeddings)


def _extract_word_embeddings_batched(
    features: torch.Tensor,
    word_mlp: base.CrissCrossWordEmbeddingExtractor,
    subsegment_info: List[Dict[str, int]],
    downsample_ratio: int,
) -> torch.Tensor:
    """Pool all word windows and execute the projection MLP in one batch."""

    pooled_words = []
    for info in subsegment_info:
        start_t, end_t = base.map_raw_to_encoded_timesteps(
            int(info["start_sample"]),
            int(info["end_sample"]),
            downsample_ratio,
        )
        if end_t <= start_t:
            raise ValueError(
                f"Empty encoded word window: raw=({info['start_sample']}, "
                f"{info['end_sample']}), encoded=({start_t}, {end_t})"
            )
        pooled_words.append(
            features[int(info["batch_idx"]), :, start_t:end_t, :].mean(dim=1)
        )

    if not pooled_words:
        return features.new_empty((0, word_mlp.embed_dim))

    pooled = torch.stack(pooled_words, dim=0)
    flattened = pooled.reshape(pooled.shape[0], -1)
    return word_mlp.mlp(flattened)


def _optimized_step(
    batch: Dict[str, Any],
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: base.CrissCrossWordEmbeddingExtractor,
    vocab_embeddings_device: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
    downsample_ratio: int,
    mixed_precision: bool,
    amp_dtype: str,
    features_only_forward: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = torch.device(device).type == "cuda"
    meg = batch["meg"].to(device, non_blocking=non_blocking)
    word_labels = batch["word_labels"].to(device, non_blocking=non_blocking)
    sensor_xyzdir = batch["sensor_xyzdir"].to(device, non_blocking=non_blocking)
    sensor_types = batch["sensor_types"].to(device, non_blocking=non_blocking)
    sensor_mask = batch["sensor_mask"].to(device, non_blocking=non_blocking)
    sensor_xyz = sensor_xyzdir[..., :3]
    sensor_abc = sensor_xyzdir[..., 3:]

    with _cuda_autocast_context(device, mixed_precision, amp_dtype):
        if features_only_forward:
            features = _feature_only_forward(
                criss_cross_model,
                meg,
                sensor_xyz,
                sensor_abc,
                sensor_types,
            )
        else:
            output = criss_cross_model(
                meg,
                sensor_xyz,
                sensor_abc,
                sensor_types,
                sensor_mask,
                apply_mask=False,
            )
            features = output["features"]

        word_embeddings = _extract_word_embeddings_batched(
            features,
            word_mlp,
            batch["subsegment_info"],
            downsample_ratio,
        )
        target_embeddings = vocab_embeddings_device.index_select(0, word_labels)
        loss = criterion(
            word_embeddings,
            target_embeddings,
            reweigh_positives=True,
        )

    # Metrics are accumulated in float32. The loss keeps its original graph for
    # backward; these casts are only applied to the returned embeddings.
    return loss, word_embeddings.float(), target_embeddings.float()


def _empty_eval_metrics(
    retrieval_set_sizes: List[int],
    k: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "loss": 0.0,
        "mean_cosine_similarity": 0.0,
        "std_cosine_similarity": 0.0,
        "mean_pred_norm": 0.0,
        "std_pred_norm": 0.0,
        "mean_target_norm": 0.0,
    }
    for retrieval_size in retrieval_set_sizes:
        metrics[f"top{k}_accuracy_retrieval{retrieval_size}"] = 0.0
        metrics[f"balanced_top{k}_accuracy_retrieval{retrieval_size}"] = 0.0
        metrics[f"n_samples_retrieval{retrieval_size}"] = 0
        metrics[f"n_skipped_retrieval{retrieval_size}"] = 0
    return metrics


def _evaluate_epoch(
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: base.CrissCrossWordEmbeddingExtractor,
    dataloader: DataLoader,
    vocab_embeddings_device: torch.Tensor,
    vocab_embeddings_cpu: torch.Tensor,
    criterion: SigLipLoss,
    device: str,
    retrieval_set_sizes: List[int],
    k: int,
    downsample_ratio: int,
    mixed_precision: bool,
    amp_dtype: str,
    features_only_forward: bool,
    description: str,
) -> Dict[str, float]:
    criss_cross_model.eval()
    word_mlp.eval()

    losses: List[float] = []
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=description):
            if len(batch["word_labels"]) == 0:
                continue
            loss, pred_embs, target_embs = _optimized_step(
                batch,
                criss_cross_model,
                word_mlp,
                vocab_embeddings_device,
                criterion,
                device,
                downsample_ratio,
                mixed_precision,
                amp_dtype,
                features_only_forward,
            )
            losses.append(float(loss.item()))
            predictions.append(pred_embs.cpu())
            targets.append(target_embs.cpu())
            labels.append(batch["word_labels"].cpu())

    if not losses:
        return _empty_eval_metrics(retrieval_set_sizes, k)

    all_predictions = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    all_labels = torch.cat(labels, dim=0)

    metrics: Dict[str, float] = {"loss": sum(losses) / len(losses)}
    for retrieval_size in retrieval_set_sizes:
        metrics.update(
            base.compute_top_k_accuracy_with_retrieval_set(
                all_predictions,
                all_labels,
                vocab_embeddings_cpu,
                retrieval_set_size=retrieval_size,
                k=k,
            )
        )
        metrics[
            f"balanced_top{k}_accuracy_retrieval{retrieval_size}"
        ] = base.compute_balanced_top_k_accuracy_with_retrieval_set(
            all_predictions,
            all_labels,
            vocab_embeddings_cpu,
            retrieval_set_size=retrieval_size,
            k=k,
        )

    metrics.update(base.compute_embedding_metrics(all_predictions, all_targets))
    return metrics


def _trainable_backbone_parameters(
    model: CrissCrossTransformerModule,
    features_only_forward: bool,
) -> Tuple[List[torch.nn.Parameter], int]:
    parameters: List[torch.nn.Parameter] = []
    excluded = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if features_only_forward and name.startswith("output_head."):
            excluded += parameter.numel()
            continue
        parameters.append(parameter)
    return parameters, excluded


def _optimization_settings(cfg: DictConfig, device: str) -> Dict[str, Any]:
    mixed_precision = bool(cfg.training.get("mixed_precision", True))
    amp_dtype = str(cfg.training.get("amp_dtype", "bfloat16"))
    gradient_checkpointing = bool(cfg.training.get("gradient_checkpointing", False))
    features_only_forward = bool(cfg.training.get("features_only_forward", True))
    evaluate_test_during_training = bool(
        cfg.evaluation.get("evaluate_test_during_training", False)
    )
    matmul_precision = str(cfg.training.get("matmul_precision", "high"))
    allow_tf32 = bool(cfg.training.get("allow_tf32", True))

    if torch.device(device).type != "cuda":
        mixed_precision = False
        allow_tf32 = False

    return {
        "mixed_precision": mixed_precision,
        "amp_dtype": amp_dtype,
        "gradient_checkpointing": gradient_checkpointing,
        "features_only_forward": features_only_forward,
        "evaluate_test_during_training": evaluate_test_during_training,
        "matmul_precision": matmul_precision,
        "allow_tf32": allow_tf32,
    }


def optimized_train_and_evaluate(
    criss_cross_model: CrissCrossTransformerModule,
    word_mlp: base.CrissCrossWordEmbeddingExtractor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab_embeddings: torch.Tensor,
    cfg: DictConfig,
    device: str,
    downsample_ratio: int,
) -> Dict[str, float]:
    """Optimized replacement for the generic downstream training loop."""

    settings = _optimization_settings(cfg, device)
    torch.set_float32_matmul_precision(settings["matmul_precision"])
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = settings["allow_tf32"]
        torch.backends.cudnn.allow_tf32 = settings["allow_tf32"]
        torch.backends.cudnn.benchmark = True

    logger.info("Downstream throughput optimizations:")
    for key, value in settings.items():
        logger.info(f"  {key}: {value}")

    if settings["gradient_checkpointing"]:
        criss_cross_model.enable_gradient_checkpointing()
    else:
        logger.info("  Gradient checkpointing disabled: activations fit comfortably in VRAM")

    backbone_parameters, excluded_output_head_parameters = _trainable_backbone_parameters(
        criss_cross_model,
        settings["features_only_forward"],
    )
    logger.info(
        "  Excluded unused output-head parameters from optimizer: "
        f"{excluded_output_head_parameters:,}"
    )

    if cfg.model.train_from_scratch:
        params = [
            {"params": backbone_parameters, "lr": cfg.training.from_scratch_lr},
            {"params": word_mlp.parameters(), "lr": cfg.training.from_scratch_lr},
        ]
    else:
        params = [
            {"params": backbone_parameters, "lr": cfg.training.criss_cross_lr},
            {"params": word_mlp.parameters(), "lr": cfg.training.word_mlp_lr},
        ]

    optimizer = AdamW(params, weight_decay=cfg.training.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )
    criterion = SigLipLoss(
        norm_kind=cfg.loss.norm_kind,
        temperature=cfg.loss.temperature,
        bias=cfg.loss.bias,
        reduction=cfg.loss.reduction,
    ).to(device)

    vocab_embeddings_cpu = vocab_embeddings.detach().float().cpu()
    vocab_embeddings_device = vocab_embeddings_cpu.to(device, non_blocking=True)

    checkpoint_dir = resolve_checkpoint_dir(cfg.logging)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_top10_acc = -math.inf
    patience_counter = 0
    best_val_epoch = 0
    start_epoch = 0
    previous_val_primary_acc: Optional[float] = None

    resume_checkpoint = cfg.training.get("resume_checkpoint", None)
    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        resume_ckpt = torch.load(resume_checkpoint, map_location=device)
        criss_cross_model.load_state_dict(resume_ckpt["criss_cross_state_dict"])
        word_mlp.load_state_dict(resume_ckpt["word_mlp_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_val_top10_acc = float(
            resume_ckpt.get("best_val_top10_acc", -math.inf)
        )
        patience_counter = int(resume_ckpt.get("patience_counter", 0))
        best_val_epoch = int(resume_ckpt.get("best_val_epoch", 0))

    retrieval_set_sizes = [int(value) for value in cfg.evaluation.retrieval_set_sizes]
    primary_retrieval_size = retrieval_set_sizes[-1]
    k = int(cfg.evaluation.k)
    primary_metric_key = (
        f"balanced_top{k}_accuracy_retrieval{primary_retrieval_size}"
    )

    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        criss_cross_model.train()
        word_mlp.train()

        train_losses: List[float] = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            if len(batch["word_labels"]) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, _predictions, _targets = _optimized_step(
                batch,
                criss_cross_model,
                word_mlp,
                vocab_embeddings_device,
                criterion,
                device,
                downsample_ratio,
                settings["mixed_precision"],
                settings["amp_dtype"],
                settings["features_only_forward"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone_parameters) + list(word_mlp.parameters()),
                cfg.training.gradient_clip_val,
            )
            optimizer.step()
            train_losses.append(float(loss.item()))

            if batch_idx % int(cfg.logging.log_every_n_steps) == 0:
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/step": epoch * len(train_loader) + batch_idx,
                    }
                )

        if not train_losses:
            raise RuntimeError("No valid training batches were produced")
        train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"  Train loss: {train_loss:.4f}")

        val_metrics = _evaluate_epoch(
            criss_cross_model,
            word_mlp,
            val_loader,
            vocab_embeddings_device,
            vocab_embeddings_cpu,
            criterion,
            device,
            retrieval_set_sizes,
            k,
            downsample_ratio,
            settings["mixed_precision"],
            settings["amp_dtype"],
            settings["features_only_forward"],
            "Validation",
        )

        test_metrics_during_training: Dict[str, float] = {}
        if settings["evaluate_test_during_training"]:
            test_metrics_during_training = _evaluate_epoch(
                criss_cross_model,
                word_mlp,
                test_loader,
                vocab_embeddings_device,
                vocab_embeddings_cpu,
                criterion,
                device,
                retrieval_set_sizes,
                k,
                downsample_ratio,
                settings["mixed_precision"],
                settings["amp_dtype"],
                settings["features_only_forward"],
                "Test during training",
            )

        val_primary_acc = float(val_metrics.get(primary_metric_key, 0.0))
        previous_best = best_val_top10_acc
        primary_delta = (
            None
            if previous_val_primary_acc is None
            else val_primary_acc - previous_val_primary_acc
        )
        primary_margin_over_best = val_primary_acc - previous_best
        improved = val_primary_acc > previous_best + float(cfg.training.min_delta)
        primary_gain_over_best = primary_margin_over_best if improved else 0.0

        logger.info(f"  Val loss: {val_metrics['loss']:.4f}")
        for retrieval_size in retrieval_set_sizes:
            val_acc = val_metrics.get(
                f"top{k}_accuracy_retrieval{retrieval_size}", 0.0
            )
            val_balanced = val_metrics.get(
                f"balanced_top{k}_accuracy_retrieval{retrieval_size}", 0.0
            )
            val_n = val_metrics.get(
                f"n_samples_retrieval{retrieval_size}", 0
            )
            logger.info(
                f"  [Retrieval {retrieval_size}] Val top-{k}: {val_acc:.4f}, "
                f"balanced: {val_balanced:.4f} (n={val_n})"
            )

        log_dict: Dict[str, Any] = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/primary_metric_value": val_primary_acc,
            "val/primary_metric_margin_over_previous_best": primary_margin_over_best,
            "val/primary_metric_gain_over_previous_best": primary_gain_over_best,
            "val/primary_metric_improved": int(improved),
            **{f"val/{name}": value for name, value in val_metrics.items()},
        }
        if primary_delta is not None:
            log_dict["val/primary_metric_delta_from_previous_epoch"] = primary_delta
        if test_metrics_during_training:
            log_dict.update(
                {
                    f"test_during_train/{name}": value
                    for name, value in test_metrics_during_training.items()
                }
            )
        wandb.log(log_dict)
        scheduler.step(val_primary_acc)

        if improved:
            best_val_top10_acc = val_primary_acc
            patience_counter = 0
            best_val_epoch = epoch + 1
            best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "criss_cross_state_dict": criss_cross_model.state_dict(),
                    "word_mlp_state_dict": word_mlp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_metrics": val_metrics,
                    "test_metrics_at_best_val": test_metrics_during_training,
                    "best_val_top10_acc": best_val_top10_acc,
                    "patience_counter": patience_counter,
                    "best_val_epoch": best_val_epoch,
                    "best_test_metrics_at_best_val": test_metrics_during_training,
                    "optimization_settings": settings,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                best_checkpoint_path,
            )
            logger.info(
                f"  Saved best model (val balanced: {best_val_top10_acc:.4f})"
            )
        else:
            patience_counter += 1

        latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "criss_cross_state_dict": criss_cross_model.state_dict(),
                "word_mlp_state_dict": word_mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "best_val_top10_acc": best_val_top10_acc,
                "patience_counter": patience_counter,
                "best_val_epoch": best_val_epoch,
                "best_test_metrics_at_best_val": test_metrics_during_training,
                "optimization_settings": settings,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            latest_checkpoint_path,
        )

        history_row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "primary_metric": primary_metric_key,
            "train/loss": train_loss,
            "val/primary_metric_value": val_primary_acc,
            "val/primary_metric_delta_from_previous_epoch": primary_delta,
            "val/primary_metric_margin_over_previous_best": primary_margin_over_best,
            "val/primary_metric_gain_over_previous_best": primary_gain_over_best,
            "val/primary_metric_improved": improved,
            "val/best_primary_metric": best_val_top10_acc,
            "val/best_epoch": best_val_epoch,
            "training/patience_counter": patience_counter,
            "training/early_stopped": patience_counter >= int(cfg.training.patience),
            **{
                f"optimizer/lr_group_{index}": group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            **{f"val/{name}": value for name, value in val_metrics.items()},
        }
        if test_metrics_during_training:
            history_row.update(
                {
                    f"test_during_train/{name}": value
                    for name, value in test_metrics_during_training.items()
                }
            )
        csv_path, jsonl_path = append_epoch_metrics_history(
            cfg.logging.save_dir,
            history_row,
            reset=(epoch == start_epoch and start_epoch == 0),
        )
        logger.info(f"  Metrics history updated: {csv_path} and {jsonl_path}")
        previous_val_primary_acc = val_primary_acc

        if patience_counter >= int(cfg.training.patience):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    best_checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint was created at {best_checkpoint_path}"
        )

    logger.info("\nLoading best validation checkpoint for the only final test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    criss_cross_model.load_state_dict(checkpoint["criss_cross_state_dict"])
    word_mlp.load_state_dict(checkpoint["word_mlp_state_dict"])

    final_test_metrics = _evaluate_epoch(
        criss_cross_model,
        word_mlp,
        test_loader,
        vocab_embeddings_device,
        vocab_embeddings_cpu,
        criterion,
        device,
        retrieval_set_sizes,
        k,
        downsample_ratio,
        settings["mixed_precision"],
        settings["amp_dtype"],
        settings["features_only_forward"],
        "Final test",
    )

    logger.info("\n=== Final Test Results ===")
    logger.info(f"Best validation epoch: {checkpoint.get('best_val_epoch', best_val_epoch)}")
    for metric_name, value in final_test_metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    wandb.log(
        {
            "best_val_epoch": checkpoint.get("best_val_epoch", best_val_epoch),
            **{
                f"test_after_load/{name}": value
                for name, value in final_test_metrics.items()
            },
        }
    )
    return final_test_metrics


def install_optimized_word_finetuning(evaluator_module=base) -> None:
    """Install the optimized loop into the generic Hydra evaluator module."""

    evaluator_module.train_and_evaluate = optimized_train_and_evaluate
