"""Two-stage EEG curriculum training entrypoint.

This reuses the proven continuous EEG training loop while replacing only the
DataModule and the EEG sensor-type embedding lookup.
"""

from brainstorm import train_criss_cross_eeg_continuous as _base
from brainstorm.data.eeg_curriculum_datamodule import CurriculumEEGDataModule
from brainstorm.models.eeg_sensor_embedding_transformer import (
    EEGSensorEmbeddingCrissCrossTransformerModule,
)


_base.MultiEEGDataModule = CurriculumEEGDataModule
_base.CrissCrossTransformerModule = EEGSensorEmbeddingCrissCrossTransformerModule


if __name__ == "__main__":
    _base.main()
