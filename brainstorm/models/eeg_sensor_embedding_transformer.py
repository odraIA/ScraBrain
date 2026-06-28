"""EEG-only Criss-Cross model with a configurable sensor-type embedding row.

The physical modality remains EEG (sensor type 2), so the base model keeps the
MEG orientation contribution disabled. Only the lookup used by
``sensor_type_layer`` can be redirected to row 1 (MEG magnetometer) or row 2
(dedicated EEG).
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

from brainstorm.models.criss_cross_transformer import (
    EEG_SENSOR_TYPE_ID,
    CrissCrossTransformerModule,
)


class EEGSensorEmbeddingCrissCrossTransformerModule(CrissCrossTransformerModule):
    """Criss-Cross Transformer with an explicit EEG type-embedding choice."""

    def __init__(
        self,
        *args,
        eeg_sensor_embedding_type_id: int | None = None,
        **kwargs,
    ) -> None:
        if eeg_sensor_embedding_type_id is None:
            eeg_sensor_embedding_type_id = int(
                os.environ.get(
                    "EEG_SENSOR_EMBEDDING_TYPE_ID",
                    str(EEG_SENSOR_TYPE_ID),
                )
            )

        super().__init__(*args, **kwargs)
        self.eeg_sensor_embedding_type_id = int(eeg_sensor_embedding_type_id)
        if not 0 <= self.eeg_sensor_embedding_type_id < self.num_sensor_types:
            raise ValueError(
                "eeg_sensor_embedding_type_id must be inside the sensor-type "
                f"vocabulary; got {self.eeg_sensor_embedding_type_id} for "
                f"num_sensor_types={self.num_sensor_types}."
            )

        print(
            "✓ EEG physical sensor type remains 2; sensor embedding lookup uses "
            f"row {self.eeg_sensor_embedding_type_id}"
        )

    def _construct_embeddings(
        self,
        codes: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings, reordered_codes = super()._construct_embeddings(
            codes,
            sensor_xyz,
            sensor_abc,
            sensor_type,
        )

        if self.eeg_sensor_embedding_type_id == EEG_SENSOR_TYPE_ID:
            return embeddings, reordered_codes

        physical_type = sensor_type.long()
        original_type_embedding = self.sensor_type_layer(physical_type)
        embedding_lookup_type = physical_type.clone()
        embedding_lookup_type[physical_type == EEG_SENSOR_TYPE_ID] = (
            self.eeg_sensor_embedding_type_id
        )
        selected_type_embedding = self.sensor_type_layer(embedding_lookup_type)

        embeddings = embeddings + (
            selected_type_embedding - original_type_embedding
        ).unsqueeze(2)
        return embeddings, reordered_codes


__all__ = ["EEGSensorEmbeddingCrissCrossTransformerModule"]
