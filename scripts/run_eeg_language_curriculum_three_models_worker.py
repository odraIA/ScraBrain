#!/usr/bin/env python3
"""GPU worker for the three-model EEG language curriculum."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
from pathlib import Path
from typing import Optional


GPU = os.environ.get("EEG_WORKER_GPU", "0")
SEED = os.environ.get("EEG_SEED", "42")
CACHE = os.environ.get("EEG_CACHE_DIR", "./data/cache/eeg_preprocessed")
RUN_ROOT = Path(os.environ["EEG_PIPELINE_RUN_ROOT"])
LOG_ROOT = Path(os.environ["EEG_TRAIN_LOG_ROOT"])
CKPT_ROOT = Path(os.environ["EEG_CHECKPOINT_ROOT"])
READ_BATCH = os.environ.get("EEG_READING_BATCH_SIZE", "4")
LANG_BATCH = os.environ.get("EEG_LANGUAGE_BATCH_SIZE", "1")
WORKERS = os.environ.get("EEG_NUM_WORKERS", "6")
VAL_EVERY = os.environ.get("EEG_VAL_CHECK_INTERVAL", "500")
SAVE_EVERY = os.environ.get("EEG_CHECKPOINT_EVERY_N_TRAIN_STEPS", "5000")
MEGXL = os.environ.get(
    "CRISS_CROSS_CHECKPOINT",
    "./checkpoints/baseline/meg-xl-med.ckpt",
)
RESUME = os.environ.get("EEG_RESUME", "true").lower() == "true"
CONTINUE_ON_ERROR = os.environ.get("CONTINUE_ON_ERROR", "true").lower() == "true"

QUEUE_FILE = RUN_ROOT / "jobs.tsv"
NEXT_FILE = RUN_ROOT / "next_job.txt"
QUEUE_LOCK = RUN_ROOT / "queue.lock"
RESULTS_LOCK = RUN_ROOT / "results.lock"
RUNS_FILE = RUN_ROOT / "runs.tsv"

RUN_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT.mkdir(parents=True, exist_ok=True)


def claim_job() -> Optional[dict[str, str]]:
    with QUEUE_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        index = int(NEXT_FILE.read_text(encoding="utf-8").strip() or "0")
        with QUEUE_FILE.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if index >= len(rows):
            return None
        NEXT_FILE.write_text(f"{index + 1}\n", encoding="utf-8")
        return rows[index]


def append_result(
    *,
    label: str,
    initialization: str,
    embedding_id: int,
    status: str,
    reading_checkpoint: str = "",
    final_checkpoint: str = "",
    message: str = "",
) -> None:
    with RESULTS_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with RUNS_FILE.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    label,
                    initialization,
                    embedding_id,
                    GPU,
                    status,
                    reading_checkpoint,
                    final_checkpoint,
                    message,
                ]
            )


def completed(experiment: str) -> bool:
    path = LOG_ROOT / experiment / "final_results.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def best_checkpoint(experiment: str) -> Optional[Path]:
    directory = CKPT_ROOT / experiment
    for name in ("checkpoint_best.pt", "checkpoint_latest.pt", "last.ckpt"):
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    candidates = sorted(directory.glob("checkpoint-*.ckpt"))
    return candidates[-1] if candidates else None


def resume_checkpoint(experiment: str) -> Optional[Path]:
    directory = CKPT_ROOT / experiment
    for name in ("last.ckpt", "checkpoint_latest.pt"):
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    return None


def common_overrides(experiment: str, batch_size: str, embedding_id: int) -> list[str]:
    return [
        "data.target_sfreq=50.0",
        "model.sampling_rate=50",
        "data.l_freq=0.1",
        "data.h_freq=50.0",
        f"data.cache_dir={CACHE}",
        f"training.batch_size={batch_size}",
        f"training.num_workers={WORKERS}",
        "training.persistent_workers=true",
        "trainer.devices=1",
        "trainer.strategy=auto",
        f"trainer.val_check_interval={VAL_EVERY}",
        f"checkpoint.every_n_train_steps={SAVE_EVERY}",
        "model.tokenizer_name=biocodec",
        "model.tokenizer_variant=default",
        "model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt",
        "model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt",
        "model.tokenizer_config_path=null",
        "model.vocab_size=256",
        "model.num_quantizers=6",
        "model.num_quantizers_used=6",
        "model.tokenizer_downsample_ratio=12",
        "model.overlap_ratio=0.0",
        "model.num_sensor_types=3",
        f"+model.eeg_sensor_embedding_type_id={embedding_id}",
        f"logging.experiment_name={experiment}",
        f"logging.save_dir={LOG_ROOT}",
        f"checkpoint.save_dir={CKPT_ROOT}",
        f"seed={SEED}",
    ]


def run_stage(
    *,
    stage: str,
    config: str,
    experiment: str,
    batch_size: str,
    initialization: str,
    embedding_id: int,
    promoted_checkpoint: Optional[Path] = None,
) -> Path:
    existing = best_checkpoint(experiment)
    if completed(experiment) and existing is not None:
        print(f"REUSING completed {stage}: {experiment}")
        return existing

    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "brainstorm.train_criss_cross_eeg_curriculum",
        "--config-name",
        config,
        *common_overrides(experiment, batch_size, embedding_id),
    ]

    resume = resume_checkpoint(experiment) if RESUME else None
    if resume is not None:
        command += [
            "checkpoint.resume=true",
            f"checkpoint.resume_path={resume}",
        ]
        print(f"RESUMING {stage} from {resume}")
    else:
        command += ["checkpoint.resume=false", "checkpoint.resume_path=null"]

    if stage == "reading" and initialization == "scratch":
        command += [
            "model.train_from_scratch=true",
            "model.use_promoted_checkpoint=false",
            "model.initialize_eeg_from_meg=false",
        ]
    elif stage == "reading":
        command += [
            "model.train_from_scratch=false",
            "model.use_promoted_checkpoint=false",
            "model.initialize_eeg_from_meg=true",
            "model.eeg_meg_sensor_type_id=1",
            f"model.criss_cross_checkpoint={MEGXL}",
        ]
    else:
        if promoted_checkpoint is None or not promoted_checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing reading checkpoint for {experiment}: {promoted_checkpoint}"
            )
        command += [
            "model.train_from_scratch=false",
            "model.use_promoted_checkpoint=true",
            "model.initialize_eeg_from_meg=false",
            f"model.promoted_checkpoint={promoted_checkpoint}",
        ]

    environment = os.environ.copy()
    environment["EEG_SENSOR_EMBEDDING_TYPE_ID"] = str(embedding_id)
    print(f"STARTING {stage}: {experiment}")
    subprocess.run(command, env=environment, check=True)

    checkpoint = best_checkpoint(experiment)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint produced for {experiment}")
    return checkpoint


def run_pipeline(job: dict[str, str]) -> None:
    label = job["label"]
    initialization = job["initialization"]
    embedding_id = int(job["eeg_embedding_id"])
    reading_experiment = f"eeg_curriculum_{label}_reading_seed{SEED}"
    language_experiment = f"eeg_curriculum_{label}_language_seed{SEED}"
    pipeline_dir = RUN_ROOT / "pipelines" / label
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    reading_checkpoint = run_stage(
        stage="reading",
        config="train_criss_cross_eeg_reading_continuous",
        experiment=reading_experiment,
        batch_size=READ_BATCH,
        initialization=initialization,
        embedding_id=embedding_id,
    )
    (pipeline_dir / "reading_checkpoint_used_for_language.txt").write_text(
        f"{reading_checkpoint}\n",
        encoding="utf-8",
    )

    final_checkpoint = run_stage(
        stage="language",
        config="train_criss_cross_eeg_language_listening_continuous",
        experiment=language_experiment,
        batch_size=LANG_BATCH,
        initialization=initialization,
        embedding_id=embedding_id,
        promoted_checkpoint=reading_checkpoint,
    )
    (pipeline_dir / "final_results.txt").write_text(
        "\n".join(
            [
                f"Pipeline: {label}",
                f"Initialization: {initialization}",
                f"EEG embedding id: {embedding_id}",
                "Reading datasets: EEGDash + ZuCo",
                "Language datasets: SparrKULee + Weissbart + Alice EEG + ds007808",
                "Held out: ds004408",
                f"Reading checkpoint: {reading_checkpoint}",
                f"Final checkpoint: {final_checkpoint}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    append_result(
        label=label,
        initialization=initialization,
        embedding_id=embedding_id,
        status="OK",
        reading_checkpoint=str(reading_checkpoint),
        final_checkpoint=str(final_checkpoint),
    )


def main() -> None:
    while True:
        job = claim_job()
        if job is None:
            print(f"GPU {GPU}: queue empty")
            return
        try:
            run_pipeline(job)
        except Exception as exc:
            append_result(
                label=job["label"],
                initialization=job["initialization"],
                embedding_id=int(job["eeg_embedding_id"]),
                status="FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
            if not CONTINUE_ON_ERROR:
                raise


if __name__ == "__main__":
    main()
