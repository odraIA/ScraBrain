#!/usr/bin/env python3
"""Materialize continuous EEG preprocessing caches without loading a model.

The script composes one of the normal training configurations, creates the same
continuity-aware DataModule used by training, and stops after ``setup('fit')``.
A separate staging cache can be seeded from and atomically published to the
main cache, which makes it safe to run beside an already-started sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import h5py
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from brainstorm.data.eeg_continuous_masked_datamodule import MultiEEGDataModule


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
_REQUIRED_H5_KEYS = {"data", "sensor_xyzdir", "sensor_types", "channel_names"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        required=True,
        choices=(
            "train_criss_cross_eeg_reading_continuous",
            "train_criss_cross_eeg_listening_continuous",
        ),
    )
    parser.add_argument("--target-sfreq", type=float, required=True)
    parser.add_argument("--l-freq", type=float, required=True)
    parser.add_argument("--h-freq", type=float, required=True)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Staging cache used while preprocessing.",
    )
    parser.add_argument(
        "--main-cache-dir",
        default=None,
        help=(
            "Optional production cache. Existing readable files seed the staging "
            "cache, and newly generated files are atomically published back here."
        ),
    )
    return parser.parse_args()


def cache_file_is_readable(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as h5_file:
            return (
                _REQUIRED_H5_KEYS.issubset(h5_file.keys())
                and int(h5_file.attrs.get("n_samples", 0)) > 0
            )
    except Exception:
        return False


def seed_staging_cache(main_cache: Path, staging_cache: Path) -> int:
    """Hard-link complete production cache files into staging when possible."""
    if not main_cache.is_dir():
        return 0

    seeded = 0
    for source in main_cache.rglob("*.h5"):
        if not cache_file_is_readable(source):
            continue
        target = staging_cache / source.relative_to(main_cache)
        if cache_file_is_readable(target):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            temporary = target.with_name(
                f".{target.name}.seed-{os.getpid()}-{uuid.uuid4().hex}.tmp"
            )
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        seeded += 1
    return seeded


def publish_staging_cache(staging_cache: Path, main_cache: Path) -> int:
    """Publish complete staged files without using the loader's shared .tmp path."""
    published = 0
    for source in staging_cache.rglob("*.h5"):
        if not cache_file_is_readable(source):
            continue

        target = main_cache / source.relative_to(staging_cache)
        if cache_file_is_readable(target):
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.publish-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source, temporary)
            if not cache_file_is_readable(temporary):
                raise RuntimeError(f"Copied cache is not readable: {temporary}")
            os.replace(temporary, target)
            published += 1
        finally:
            temporary.unlink(missing_ok=True)
    return published


def compose_config(args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"data.target_sfreq={args.target_sfreq}",
        f"data.l_freq={args.l_freq}",
        f"data.h_freq={args.h_freq}",
        f"data.cache_dir={args.cache_dir}",
        f"model.sampling_rate={int(args.target_sfreq)}",
        "training.batch_size=1",
        "training.num_workers=0",
        "training.persistent_workers=false",
        "trainer.devices=1",
        "trainer.strategy=auto",
    ]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        return compose(config_name=args.config_name, overrides=overrides)


def build_datamodule(cfg: DictConfig) -> MultiEEGDataModule:
    tokenizer_name = str(cfg.model.get("tokenizer_name", "biocodec"))
    datasets_config: Any = OmegaConf.to_container(
        cfg.datasets_config,
        resolve=True,
    )
    return MultiEEGDataModule(
        datasets_config=datasets_config,
        segment_length=float(cfg.data.segment_length),
        subsegment_duration=float(cfg.data.get("subsegment_duration", 3.0)),
        words_per_segment=int(cfg.data.get("words_per_segment", 50)),
        window_onset_offset=float(cfg.data.get("window_onset_offset", -0.5)),
        cache_dir=str(cfg.data.cache_dir),
        l_freq=float(cfg.data.l_freq),
        h_freq=float(cfg.data.h_freq),
        target_sfreq=float(cfg.data.target_sfreq),
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        use_recording_sampler=bool(cfg.training.use_recording_sampler),
        sampler_seed=int(cfg.training.sampler_seed),
        debug_mode=bool(cfg.data.get("debug_mode", False)),
        max_channel_dim=cfg.data.get("max_channel_dim", None),
        infer_max_channel_dim=bool(cfg.data.get("infer_max_channel_dim", True)),
        recording_subsample_prop=cfg.data.get("recording_subsample_prop", None),
        allow_missing_word_alignment=bool(
            cfg.data.get("allow_missing_word_alignment", False)
        ),
        tokenizer_name=tokenizer_name,
    )


def dataset_length(dataset: Any) -> int:
    return len(dataset) if dataset is not None else 0


def main() -> None:
    args = parse_args()
    staging_cache = Path(args.cache_dir).resolve()
    main_cache = (
        Path(args.main_cache_dir).resolve()
        if args.main_cache_dir is not None
        else None
    )

    if main_cache is not None and main_cache == staging_cache:
        raise ValueError("Staging and main cache directories must be different")

    staging_cache.mkdir(parents=True, exist_ok=True)
    seeded = 0
    if main_cache is not None:
        main_cache.mkdir(parents=True, exist_ok=True)
        seeded = seed_staging_cache(main_cache, staging_cache)
        print(f"Seeded {seeded} readable cache files from {main_cache}")

    cfg = compose_config(args)
    print(
        "Preparing EEG cache: "
        f"config={args.config_name}, band={args.l_freq}-{args.h_freq} Hz, "
        f"target_sfreq={args.target_sfreq} Hz"
    )

    datamodule = build_datamodule(cfg)
    try:
        datamodule.setup("fit")
        summary = {
            "config_name": args.config_name,
            "target_sfreq": args.target_sfreq,
            "l_freq": args.l_freq,
            "h_freq": args.h_freq,
            "staging_cache": str(staging_cache),
            "seeded_files": seeded,
            "train_segments": dataset_length(datamodule.train_dataset),
            "validation_segments": dataset_length(datamodule.val_dataset),
        }
    finally:
        datamodule.teardown("fit")

    published = 0
    if main_cache is not None:
        published = publish_staging_cache(staging_cache, main_cache)
    summary["published_files"] = published
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
