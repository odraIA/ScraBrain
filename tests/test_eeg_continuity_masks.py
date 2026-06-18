import torch
import torch.nn as nn

from brainstorm.data.eeg_continuous_masked_dataset import (
    ContinuityAwareEEGMixin,
)
from brainstorm.models.criss_cross_transformer import (
    CrissCrossTransformerModule,
)


class DummyTokenizer(nn.Module):
    n_q = 1
    vocab_size = 8
    downsample_ratio = 2

    def __init__(self):
        super().__init__()
        self.register_buffer("codebook", torch.randn(8, 2))

    def codebook_embedding(self, quantizer_idx):
        assert quantizer_idx == 0
        return self.codebook


def make_model():
    torch.manual_seed(3)
    return CrissCrossTransformerModule(
        tokenizer=DummyTokenizer(),
        latent_dim=8,
        num_layers=1,
        num_heads=2,
        vocab_size=8,
        sampling_rate=2,
        mask_duration=2.0,
        num_subsegments_to_mask=2,
        fourier_pos_dim=4,
        num_sensor_types=3,
    ).eval()


def test_complete_run_windows_drop_incomplete_remainder():
    dataset = object.__new__(ContinuityAwareEEGMixin)
    dataset.segment_length = 10.0
    dataset.target_sfreq = 1.0
    dataset.subsegment_duration = 3.0
    dataset.segment_starts = []
    dataset.recordings = [
        {
            "total_samples": 25,
            "target_ranges": [(0, 25)],
        }
    ]

    index = ContinuityAwareEEGMixin._build_segment_index(dataset)

    assert dataset.segment_starts == [[0, 10]]
    assert index == [(0, 0), (0, 1)]


def test_targeted_mask_never_leaves_listening_or_valid_sensors():
    model = make_model()
    target_mask = torch.tensor(
        [[False, False, True, True, True, True, True, True, False, False]]
    )
    sensor_mask = torch.tensor([[True, False]])

    mask, _ = model._generate_temporal_block_mask(
        B=1,
        n_channels=2,
        n_timesteps=10,
        sensor_mask=sensor_mask,
        device=torch.device("cpu"),
        target_mask=target_mask,
    )

    assert not mask[0, 1].any()
    assert torch.all(mask[0, 0] <= target_mask[0])
    assert int(mask[0, 0].sum()) <= 2 * model.mask_length


def test_eeg_orientation_is_gated_but_meg_orientation_is_kept():
    model = make_model()
    codes = torch.zeros(1, 2, 1, 3, dtype=torch.long)
    sensor_xyz = torch.zeros(1, 2, 3)
    sensor_types = torch.tensor([[2, 0]])  # EEG, GRAD

    orientation_a = torch.zeros(1, 2, 3)
    orientation_b = orientation_a.clone()
    orientation_b[:, :, 0] = 0.75

    with torch.no_grad():
        embedded_a, _ = model._construct_embeddings(
            codes,
            sensor_xyz,
            orientation_a,
            sensor_types,
        )
        embedded_b, _ = model._construct_embeddings(
            codes,
            sensor_xyz,
            orientation_b,
            sensor_types,
        )

    assert torch.allclose(embedded_a[:, 0], embedded_b[:, 0])
    assert not torch.allclose(embedded_a[:, 1], embedded_b[:, 1])


def test_meg_batch_keeps_original_five_item_interface():
    batch = (
        torch.randn(2, 4, 20),
        torch.randn(2, 4, 6),
        torch.zeros(2, 4, dtype=torch.long),
        torch.ones(2, 4),
        torch.tensor([0, 1]),
    )

    unpacked = CrissCrossTransformerModule._unpack_batch(batch)

    assert unpacked[4] is None
    assert torch.equal(unpacked[5], batch[4])
