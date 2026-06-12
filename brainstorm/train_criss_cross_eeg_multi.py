"""Training script for Criss-Cross Transformer on multiple OpenNeuro EEG datasets.

This is the EEG counterpart of ``train_criss_cross_multi.py``. It keeps the
same model and training loop style, but uses ``MultiEEGDataModule`` and writes a
self-contained run folder with:

- config_resolved.yaml
- stdout_stderr.log
- epoch_metrics.csv / epoch_metrics.jsonl
- final_results.txt / final_results.json
- checkpoint_latest.pt / checkpoint_best.pt copies when available
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brainstorm.data.eeg_multi_datamodule import MultiEEGDataModule
from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.neuro_tokenizers.factory import load_neuro_tokenizer


class TeeStream:
    """Mirror stdout/stderr to a file while keeping terminal output."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return None


def git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8").strip()
    except Exception:
        return None


def install_tee(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = (save_dir / "stdout_stderr.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_file


class SamplerVerificationCallback(Callback):
    """Lightweight check that the recording sampler is being used as expected."""

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch > 0:
            return
        sampler = trainer.train_dataloader.sampler
        try:
            from brainstorm.data.samplers import RecordingShuffleSampler
        except Exception:
            print("Sampler check skipped: could not import RecordingShuffleSampler")
            return

        if trainer.world_size > 1:
            underlying = getattr(sampler, "sampler", sampler)
            if isinstance(underlying, RecordingShuffleSampler):
                print(
                    f"✓ Rank {trainer.global_rank}/{trainer.world_size}: "
                    f"RecordingShuffleSampler wrapped by {type(sampler).__name__}"
                )
            else:
                print(
                    f"⚠ Rank {trainer.global_rank}: expected RecordingShuffleSampler, "
                    f"got {type(underlying).__name__}"
                )
        else:
            if isinstance(sampler, RecordingShuffleSampler):
                print("✓ Single GPU: using RecordingShuffleSampler")
            else:
                print(f"⚠ Expected RecordingShuffleSampler, got {type(sampler).__name__}")


class MetricsFileCallback(Callback):
    """Persist Lightning callback metrics to JSONL and CSV after validation."""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.jsonl_path = save_dir / "epoch_metrics.jsonl"
        self.csv_path = save_dir / "epoch_metrics.csv"
        self.rows: list[dict[str, Any]] = []

    def _serialize_metrics(self, trainer, stage: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "timestamp": now_iso(),
            "stage": stage,
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }
        for key, value in trainer.callback_metrics.items():
            scalar = safe_float(value)
            if scalar is not None:
                row[str(key)] = scalar
        return row

    def _write(self, row: dict[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        self.rows.append(row)
        fieldnames = sorted({key for item in self.rows for key in item.keys()})
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in self.rows:
                writer.writerow(item)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._write(self._serialize_metrics(trainer, "validation"))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._write(self._serialize_metrics(trainer, "train"))


def resolve_save_dir(cfg: DictConfig) -> Path:
    base = Path(str(cfg.logging.get("save_dir", "./logs/eeg_multi_training")))
    exp = str(cfg.logging.get("experiment_name", "eeg_multi_training"))
    return base / exp


def write_config_snapshot(save_dir: Path, cfg: DictConfig) -> Path:
    path = save_dir / "config_resolved.yaml"
    path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def load_partial_checkpoint(model: torch.nn.Module, checkpoint_path: str | None) -> dict[str, Any]:
    if not checkpoint_path:
        return {"requested": False, "loaded": False, "reason": "no checkpoint path"}

    path = Path(checkpoint_path)
    if not path.exists():
        return {"requested": True, "loaded": False, "reason": f"checkpoint not found: {path}"}

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    current = model.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    skipped_shape = []
    skipped_missing = []
    for key, value in state_dict.items():
        if key not in current:
            skipped_missing.append(key)
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped_shape.append({"key": key, "checkpoint": list(value.shape), "model": list(current[key].shape)})
            continue
        compatible[key] = value

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "requested": True,
        "loaded": True,
        "checkpoint": str(path),
        "loaded_keys": len(compatible),
        "total_checkpoint_keys": len(state_dict),
        "missing_after_load": list(missing),
        "unexpected_after_load": list(unexpected),
        "skipped_missing_count": len(skipped_missing),
        "skipped_shape_count": len(skipped_shape),
        "skipped_shape_examples": skipped_shape[:20],
    }


def copy_if_exists(src: str | Path | None, dst: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def write_final_results(
    *,
    save_dir: Path,
    checkpoint_dir: Path,
    cfg: DictConfig,
    status: str,
    error: Optional[str],
    checkpoint_callback: Optional[ModelCheckpoint],
    checkpoint_load_report: dict[str, Any],
    config_snapshot: Path,
    metrics_callback: MetricsFileCallback,
    tokenizer: Any,
    datamodule: Optional[MultiEEGDataModule],
) -> None:
    best_model_path = None
    best_model_score = None
    last_model_path = None
    if checkpoint_callback is not None:
        best_model_path = checkpoint_callback.best_model_path or None
        best_model_score = safe_float(checkpoint_callback.best_model_score)
        last_model_path = checkpoint_callback.last_model_path or None

    latest_copy = copy_if_exists(last_model_path, checkpoint_dir / "checkpoint_latest.pt")
    best_copy = copy_if_exists(best_model_path, checkpoint_dir / "checkpoint_best.pt")

    split_sizes: dict[str, int] = {}
    if datamodule is not None:
        for attr, name in (("train_dataset", "train"), ("val_dataset", "validation")):
            dataset = getattr(datamodule, attr, None)
            if dataset is not None:
                try:
                    split_sizes[name] = len(dataset)
                except Exception:
                    pass

    final_payload = {
        "status": status,
        "error": error,
        "timestamp": now_iso(),
        "git_hash": git_hash(),
        "experiment_name": str(cfg.logging.get("experiment_name", "")),
        "frequency": {
            "target_sfreq": float(cfg.data.target_sfreq),
            "l_freq": float(cfg.data.l_freq),
            "h_freq": float(cfg.data.h_freq),
        },
        "tokenizer": {
            "name": str(cfg.model.get("tokenizer_name", "biocodec")),
            "variant": str(cfg.model.get("tokenizer_variant", "")),
            "checkpoint": str(cfg.model.get("tokenizer_checkpoint", cfg.model.get("tokenizer_ckpt", ""))),
            "downsample_ratio": int(getattr(tokenizer, "downsample_ratio", 0)),
            "n_q": int(getattr(tokenizer, "n_q", 0)),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        },
        "initialization": {
            "train_from_scratch": bool(cfg.model.get("train_from_scratch", True)),
            "use_promoted_checkpoint": bool(cfg.model.get("use_promoted_checkpoint", False)),
            "criss_cross_checkpoint": str(cfg.model.get("criss_cross_checkpoint", "")),
            "promoted_checkpoint": str(cfg.model.get("promoted_checkpoint", "")),
        },
        "training": {
            "num_epochs": cfg.training.get("num_epochs", None),
            "max_steps": cfg.training.get("max_steps", None),
            "batch_size": int(cfg.training.batch_size),
            "learning_rate": float(cfg.training.learning_rate),
        },
        "split_sizes": split_sizes,
        "checkpoint_load_report": checkpoint_load_report,
        "paths": {
            "run_dir": str(save_dir),
            "config_resolved": str(config_snapshot),
            "stdout_stderr": str(save_dir / "stdout_stderr.log"),
            "epoch_metrics_csv": str(metrics_callback.csv_path),
            "epoch_metrics_jsonl": str(metrics_callback.jsonl_path),
            "checkpoint_dir": str(checkpoint_dir),
            "lightning_best_checkpoint": str(best_model_path or ""),
            "lightning_last_checkpoint": str(last_model_path or ""),
            "checkpoint_best_copy": str(best_copy or ""),
            "checkpoint_latest_copy": str(latest_copy or ""),
        },
        "best_validation_loss": best_model_score,
        "command": [sys.executable, *sys.argv],
    }

    (save_dir / "final_results.json").write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    lines = [
        "EEG Multi-Dataset Criss-Cross Training Results",
        "=" * 55,
        f"Status: {status}",
        f"Experiment: {final_payload['experiment_name']}",
        f"Timestamp: {final_payload['timestamp']}",
        f"Git hash: {final_payload['git_hash']}",
        "",
        "Frequency / preprocessing:",
        f"  target_sfreq: {final_payload['frequency']['target_sfreq']}",
        f"  l_freq:       {final_payload['frequency']['l_freq']}",
        f"  h_freq:       {final_payload['frequency']['h_freq']}",
        "",
        "Tokenizer:",
        f"  name:             {final_payload['tokenizer']['name']}",
        f"  variant:          {final_payload['tokenizer']['variant']}",
        f"  checkpoint:       {final_payload['tokenizer']['checkpoint']}",
        f"  downsample_ratio: {final_payload['tokenizer']['downsample_ratio']}",
        f"  n_q:              {final_payload['tokenizer']['n_q']}",
        f"  vocab_size:       {final_payload['tokenizer']['vocab_size']}",
        "",
        "Initialization:",
        f"  train_from_scratch:      {final_payload['initialization']['train_from_scratch']}",
        f"  use_promoted_checkpoint: {final_payload['initialization']['use_promoted_checkpoint']}",
        f"  criss_cross_checkpoint:  {final_payload['initialization']['criss_cross_checkpoint']}",
        f"  promoted_checkpoint:     {final_payload['initialization']['promoted_checkpoint']}",
        "",
        "Best checkpoint:",
        f"  best_validation_loss: {best_model_score}",
        f"  lightning_best:       {best_model_path}",
        f"  copied_best:          {best_copy}",
        f"  copied_latest:        {latest_copy}",
        "",
        "Where to look:",
        f"  Run directory:     {save_dir}",
        f"  Console log:       {save_dir / 'stdout_stderr.log'}",
        f"  Epoch metrics CSV: {metrics_callback.csv_path}",
        f"  Epoch metrics JSONL:{metrics_callback.jsonl_path}",
        f"  Checkpoints:       {checkpoint_dir}",
        f"  Full JSON summary: {save_dir / 'final_results.json'}",
    ]
    if error:
        lines.extend(["", "Error:", error])
    (save_dir / "final_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@hydra.main(version_base=None, config_path="../configs", config_name="train_criss_cross_eeg_multi_feq")
def main(cfg: DictConfig):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    save_dir = resolve_save_dir(cfg)
    log_file = install_tee(save_dir)
    config_snapshot = write_config_snapshot(save_dir, cfg)

    checkpoint_dir = Path(str(cfg.checkpoint.get("save_dir", "./checkpoints"))) / str(
        cfg.logging.get("experiment_name", "eeg_multi_training")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback: Optional[ModelCheckpoint] = None
    metrics_callback = MetricsFileCallback(save_dir)
    tokenizer = None
    datamodule: Optional[MultiEEGDataModule] = None
    checkpoint_load_report: dict[str, Any] = {"requested": False, "loaded": False}

    status = "failed"
    error_text: Optional[str] = None

    try:
        print("\n" + "=" * 80)
        print("MULTI-DATASET EEG CRISS-CROSS TRANSFORMER TRAINING")
        print("=" * 80)
        print("\n=== Configuration ===")
        print(OmegaConf.to_yaml(cfg))

        if float(cfg.data.target_sfreq) != float(cfg.model.sampling_rate):
            raise ValueError(
                f"data.target_sfreq ({cfg.data.target_sfreq}) must match "
                f"model.sampling_rate ({cfg.model.sampling_rate})."
            )
        if float(cfg.data.h_freq) >= float(cfg.data.target_sfreq) / 2.0:
            raise ValueError(
                f"Invalid filter: h_freq={cfg.data.h_freq} must be lower than "
                f"Nyquist={float(cfg.data.target_sfreq) / 2.0}."
            )
        tokenizer_name = str(cfg.model.get("tokenizer_name", "biocodec"))
        print("✓ Config validation passed")

        torch.set_float32_matmul_precision("high")
        if hasattr(cfg, "seed"):
            pl.seed_everything(int(cfg.seed), workers=True)
            print(f"✓ Random seed set to {cfg.seed}")

        print("\n" + "=" * 80)
        print("SETTING UP EEG DATA")
        print("=" * 80)
        datamodule = MultiEEGDataModule(
            datasets_config=OmegaConf.to_container(cfg.datasets_config, resolve=True),
            segment_length=float(cfg.data.segment_length),
            subsegment_duration=float(cfg.data.get("subsegment_duration", 3.0)),
            words_per_segment=int(cfg.data.get("words_per_segment", 50)),
            window_onset_offset=float(cfg.data.get("window_onset_offset", -0.5)),
            cache_dir=str(cfg.data.cache_dir),
            l_freq=float(cfg.data.l_freq),
            h_freq=float(cfg.data.h_freq),
            target_sfreq=float(cfg.data.target_sfreq),
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.num_workers),
            pin_memory=bool(cfg.training.pin_memory),
            persistent_workers=bool(cfg.training.persistent_workers),
            use_recording_sampler=bool(cfg.training.use_recording_sampler),
            sampler_seed=int(cfg.training.sampler_seed),
            debug_mode=bool(cfg.data.get("debug_mode", False)),
            max_channel_dim=cfg.data.get("max_channel_dim", None),
            infer_max_channel_dim=bool(cfg.data.get("infer_max_channel_dim", True)),
            recording_subsample_prop=cfg.data.get("recording_subsample_prop", None),
            allow_missing_word_alignment=bool(cfg.data.get("allow_missing_word_alignment", False)),
            tokenizer_name=tokenizer_name,
        )
        datamodule.setup("fit")

        num_epochs = cfg.training.get("num_epochs", None)
        max_steps = cfg.training.get("max_steps", None)
        if num_epochs is not None and max_steps is not None:
            raise ValueError("Set only one of training.num_epochs or training.max_steps.")
        if num_epochs is None and max_steps is None:
            raise ValueError("Set either training.num_epochs or training.max_steps.")

        steps_per_epoch = len(datamodule.train_dataloader())
        training_steps = int(max_steps) if max_steps is not None else int(num_epochs) * steps_per_epoch
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {training_steps}")

        print("\n" + "=" * 80)
        print("LOADING TOKENIZER")
        print("=" * 80)
        tokenizer_checkpoint = cfg.model.get("tokenizer_checkpoint", cfg.model.get("tokenizer_ckpt", None))
        print(f"Tokenizer name: {tokenizer_name}")
        print(f"Tokenizer checkpoint: {tokenizer_checkpoint}")
        tokenizer = load_neuro_tokenizer(
            tokenizer_name=tokenizer_name,
            checkpoint_path=tokenizer_checkpoint,
            device="cpu",
        )
        print("✓ Tokenizer loaded")
        print(f"  RVQ levels: {tokenizer.n_q}")
        print(f"  Codebook size: {tokenizer.vocab_size}")
        print(f"  Downsample ratio: {tokenizer.downsample_ratio}")

        print("\n" + "=" * 80)
        print("CREATING MODEL")
        print("=" * 80)
        model = CrissCrossTransformerModule(
            tokenizer=tokenizer,
            latent_dim=int(cfg.model.latent_dim),
            num_layers=int(cfg.model.num_layers),
            num_heads=int(cfg.model.num_heads),
            vocab_size=int(cfg.model.vocab_size),
            learning_rate=float(cfg.training.learning_rate),
            warmup_steps=int(cfg.training.warmup_steps),
            training_steps=training_steps,
            mask_duration=float(cfg.model.get("mask_duration", 3.0)),
            num_subsegments_to_mask=int(cfg.model.get("num_subsegments_to_mask", 20)),
            sampling_rate=int(cfg.model.sampling_rate),
            fourier_pos_dim=int(cfg.model.get("fourier_pos_dim", 250)),
            num_sensor_types=int(cfg.model.get("num_sensor_types", 3)),
        )
        if bool(cfg.model.get("use_gradient_checkpointing", False)):
            model.enable_gradient_checkpointing()

        train_from_scratch = bool(cfg.model.get("train_from_scratch", True))
        init_checkpoint = cfg.model.get("promoted_checkpoint", None) if bool(cfg.model.get("use_promoted_checkpoint", False)) else cfg.model.get("criss_cross_checkpoint", None)
        if train_from_scratch:
            checkpoint_load_report = {"requested": False, "loaded": False, "reason": "model.train_from_scratch=true"}
            print("✓ Training from scratch")
        else:
            checkpoint_load_report = load_partial_checkpoint(model, init_checkpoint)
            print("Checkpoint load report:")
            print(json.dumps(checkpoint_load_report, indent=2)[:5000])

        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        wandb_logger = None
        loggers = []
        if str(cfg.logging.get("wandb_project", "")):
            wandb_logger = WandbLogger(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.get("wandb_entity", None),
                name=cfg.logging.experiment_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                save_dir=str(save_dir),
            )
            loggers.append(wandb_logger)
            print(f"✓ WandB logger: project={cfg.logging.wandb_project}")

        csv_logger = CSVLogger(save_dir=str(save_dir), name="lightning_csv")
        loggers.append(csv_logger)
        print(f"✓ Local CSV logger: {save_dir / 'lightning_csv'}")

        callbacks = [LearningRateMonitor(logging_interval="step"), SamplerVerificationCallback(), metrics_callback]
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="checkpoint-{epoch:02d}-{step:06d}",
            monitor=str(cfg.checkpoint.get("monitor", "val/loss")),
            mode=str(cfg.checkpoint.get("mode", "min")),
            every_n_train_steps=cfg.checkpoint.get("every_n_train_steps", None),
            save_top_k=int(cfg.checkpoint.get("save_top_k", 1)),
            save_last=bool(cfg.checkpoint.get("save_last", True)),
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        trainer_kwargs = {
            "accelerator": cfg.trainer.accelerator,
            "devices": cfg.trainer.devices,
            "precision": cfg.trainer.precision,
            "callbacks": callbacks,
            "logger": loggers,
            "gradient_clip_val": float(cfg.training.gradient_clip_val),
            "log_every_n_steps": int(cfg.logging.log_every_n_steps),
            "accumulate_grad_batches": int(cfg.trainer.accumulate_grad_batches),
            "val_check_interval": cfg.trainer.val_check_interval,
            "deterministic": "warn" if hasattr(cfg, "seed") else False,
        }
        if cfg.trainer.get("strategy", None) is not None:
            trainer_kwargs["strategy"] = cfg.trainer.strategy
        if num_epochs is not None:
            trainer_kwargs["max_epochs"] = int(num_epochs)
        else:
            trainer_kwargs["max_steps"] = int(max_steps)

        trainer = pl.Trainer(**trainer_kwargs)

        ckpt_path = None
        if bool(cfg.checkpoint.get("resume", False)):
            ckpt_path = cfg.checkpoint.get("resume_path", None)
            if not ckpt_path:
                raise ValueError("checkpoint.resume=true but checkpoint.resume_path is empty")

        print("\n" + "=" * 80)
        print("STARTING EEG MULTI-DATASET TRAINING")
        print("=" * 80)
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        status = "completed"

    except KeyboardInterrupt:
        status = "interrupted"
        error_text = "Training interrupted by user."
        print("\nTRAINING INTERRUPTED")
    except Exception:
        status = "failed"
        error_text = traceback.format_exc()
        print(error_text)
        raise
    finally:
        try:
            write_final_results(
                save_dir=save_dir,
                checkpoint_dir=checkpoint_dir,
                cfg=cfg,
                status=status,
                error=error_text,
                checkpoint_callback=checkpoint_callback,
                checkpoint_load_report=checkpoint_load_report,
                config_snapshot=config_snapshot,
                metrics_callback=metrics_callback,
                tokenizer=tokenizer if tokenizer is not None else object(),
                datamodule=datamodule,
            )
        finally:
            if datamodule is not None:
                datamodule.teardown("fit")
            if 'wandb_logger' in locals() and wandb_logger is not None:
                wandb_logger.experiment.finish()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
