"""Run the EEG trainer with continuity-aware data and model classes.

The shared trainer imports the legacy DataModule and Criss-Cross model modules.
This entrypoint swaps only those exported classes before importing the trainer,
so the original MEG-XL training command remains unchanged.
"""

from __future__ import annotations

import brainstorm.data.eeg_multi_datamodule as legacy_datamodule
import brainstorm.models.criss_cross_transformer as legacy_model
from brainstorm.data.eeg_continuous_masked_datamodule import MultiEEGDataModule
from brainstorm.models.eeg_criss_cross_transformer import (
    CrissCrossTransformerModule,
)

legacy_datamodule.MultiEEGDataModule = MultiEEGDataModule
legacy_model.CrissCrossTransformerModule = CrissCrossTransformerModule

from brainstorm.train_criss_cross_eeg_multi import main  # noqa: E402


if __name__ == "__main__":
    main()
