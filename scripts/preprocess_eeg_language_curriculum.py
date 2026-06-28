#!/usr/bin/env python3
"""Preprocess caches for the reading -> language-listening curriculum."""

from __future__ import annotations

import argparse

import preprocess_eeg_reading_listening as _base

from brainstorm.data.eeg_curriculum_datamodule import CurriculumEEGDataModule


_CONFIGS = (
    "train_criss_cross_eeg_reading_continuous",
    "train_criss_cross_eeg_language_listening_continuous",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, choices=_CONFIGS)
    parser.add_argument("--target-sfreq", type=float, required=True)
    parser.add_argument("--l-freq", type=float, required=True)
    parser.add_argument("--h-freq", type=float, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--main-cache-dir", default=None)
    return parser.parse_args()


_base.MultiEEGDataModule = CurriculumEEGDataModule
_base.parse_args = parse_args


if __name__ == "__main__":
    _base.main()
