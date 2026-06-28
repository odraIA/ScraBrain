#!/usr/bin/env python3
"""Run exactly one queued curriculum pipeline in the assigned container."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_eeg_language_curriculum_three_models_worker import (  # noqa: E402
    RUN_ROOT,
    append_result,
    claim_job,
    run_pipeline,
)


def correct_dataset_manifest(label: str) -> None:
    path = RUN_ROOT / "pipelines" / label / "final_results.txt"
    if not path.is_file():
        return

    content = path.read_text(encoding="utf-8")
    content = content.replace(
        "Language datasets: SparrKULee + Weissbart + Alice EEG + ds007808",
        "Language datasets: SparrKULee + ds007808",
    )
    if "Excluded from language stage:" not in content:
        content = content.replace(
            "Held out: ds004408",
            "Excluded from language stage: Alice EEG + Weissbart EEG\n"
            "Held out: ds004408",
        )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    job = claim_job()
    if job is None:
        print("Queue is empty")
        return

    try:
        run_pipeline(job)
        correct_dataset_manifest(job["label"])
    except Exception as exc:
        append_result(
            label=job["label"],
            initialization=job["initialization"],
            embedding_id=int(job["eeg_embedding_id"]),
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
