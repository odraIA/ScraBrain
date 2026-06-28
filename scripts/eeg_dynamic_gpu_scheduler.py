#!/usr/bin/env python3
"""Launch each curriculum pipeline on the first GPU that becomes free."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import subprocess
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.environ.get("EEG_GPU_POLL_SECONDS", "30")),
    )
    parser.add_argument(
        "--max-used-memory-mb",
        type=int,
        default=int(os.environ.get("EEG_GPU_MAX_USED_MEMORY_MB", "1024")),
    )
    parser.add_argument(
        "--lock-dir",
        default=os.environ.get(
            "EEG_GPU_LOCK_DIR",
            "/tmp/scrabrain-eeg-gpu-locks",
        ),
    )
    return parser.parse_args()


def run_text(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return result.stdout.strip()


def gpu_is_free(gpu: str, max_used_memory_mb: int) -> bool:
    try:
        process_output = run_text(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
        running_pids = [
            line.strip()
            for line in process_output.splitlines()
            if line.strip().isdigit()
        ]
        if running_pids:
            return False

        memory_output = run_text(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        used_memory = int(memory_output.splitlines()[0].strip())
        return used_memory <= max_used_memory_mb
    except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
        print(f"GPU {gpu}: availability check failed: {exc}", flush=True)
        return False


def read_job_count(queue_file: Path) -> int:
    with queue_file.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def write_container_logs(container_id: str, destination: Path) -> None:
    result = subprocess.run(
        ["docker", "logs", container_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    destination.write_text(result.stdout, encoding="utf-8")


def run_one_worker(
    *,
    slot: int,
    gpu: str,
    args: argparse.Namespace,
    worker_env_file: Path,
) -> int:
    name = f"scrabrain_curriculum_worker_{slot}_{args.stamp}"
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    environment = os.environ.copy()
    environment["EEG_GPU"] = gpu
    command = [
        "docker",
        "compose",
        "-f",
        args.compose_file,
        "run",
        "-d",
        "--no-deps",
        "--env-from-file",
        str(worker_env_file),
        "-e",
        f"EEG_WORKER_GPU={gpu}",
        "--name",
        name,
        "eeg_train_reading_listening",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/run_eeg_language_curriculum_single_job_worker.py",
    ]
    output = run_text(command, env=environment)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Docker did not return a container id for worker {slot}")
    container_id = lines[-1]

    (Path(args.run_root) / f"worker_{slot}.container_id").write_text(
        f"{container_id}\n",
        encoding="utf-8",
    )
    (Path(args.run_root) / f"worker_{slot}.container_name").write_text(
        f"{name}\n",
        encoding="utf-8",
    )

    wait_output = run_text(["docker", "wait", container_id])
    try:
        exit_code = int(wait_output.splitlines()[-1])
    except (ValueError, IndexError):
        exit_code = 1

    write_container_logs(
        container_id,
        Path(args.run_root) / f"worker_{slot}.container.log",
    )
    subprocess.run(
        ["docker", "rm", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return exit_code


def wait_for_gpu_and_run(
    *,
    slot: int,
    args: argparse.Namespace,
    worker_env_file: Path,
    lock_dir: Path,
) -> None:
    announced_wait = False

    while True:
        for gpu in args.gpus:
            lock_path = lock_dir / f"gpu-{gpu}.lock"
            lock_handle = lock_path.open("a+")

            try:
                fcntl.flock(
                    lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                lock_handle.close()
                continue

            if not gpu_is_free(gpu, args.max_used_memory_mb):
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
                continue

            assignment = Path(args.run_root) / f"worker_{slot}.gpu"
            assignment.write_text(f"{gpu}\n", encoding="utf-8")
            print(f"Worker {slot}: reserved free GPU {gpu}", flush=True)

            try:
                exit_code = run_one_worker(
                    slot=slot,
                    gpu=gpu,
                    args=args,
                    worker_env_file=worker_env_file,
                )
                print(
                    f"Worker {slot}: GPU {gpu} finished with exit code {exit_code}",
                    flush=True,
                )
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            return

        if not announced_wait:
            print(
                f"Worker {slot}: no GPU is free; waiting and polling "
                f"every {args.poll_seconds:g} seconds",
                flush=True,
            )
            announced_wait = True
        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    queue_file = run_root / "jobs.tsv"
    worker_env_file = run_root / "worker.env"
    lock_dir = Path(args.lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)

    job_count = read_job_count(queue_file)
    if job_count < 1:
        raise RuntimeError(f"No jobs found in {queue_file}")

    print("Building EEG training image once before scheduling jobs", flush=True)
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            args.compose_file,
            "build",
            "eeg_train_reading_listening",
        ],
        check=True,
    )

    threads = [
        threading.Thread(
            target=wait_for_gpu_and_run,
            kwargs={
                "slot": slot,
                "args": args,
                "worker_env_file": worker_env_file,
                "lock_dir": lock_dir,
            },
            name=f"gpu-waiter-{slot}",
        )
        for slot in range(job_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print("All curriculum jobs have finished", flush=True)


if __name__ == "__main__":
    main()
