"""Alice evaluator entry point with the documented missing-marker fix enabled."""

from __future__ import annotations

from brainstorm import evaluate_criss_cross_word_classification_alice_reported as base
from brainstorm.data.alice_eeg_word_aligned_dataset_missing_first_fix import (
    AliceEEGWordAlignedDatasetMissingFirstFix,
)


# ``base.get_dataset_class`` resolves this global at call time, so replacing it
# keeps the reporting/evaluation implementation unchanged while using the
# corrected dataset loader.
base.AliceEEGWordAlignedDataset = AliceEEGWordAlignedDatasetMissingFirstFix


def main():
    return base.main()


if __name__ == "__main__":
    main()
