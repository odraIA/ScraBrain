import torch

from brainstorm.models.eeg_criss_cross_transformer import (
    MaskedSpatialTemporalEncoder,
)


def test_invalid_sensor_and_time_tokens_cannot_change_valid_outputs():
    torch.manual_seed(7)
    encoder = MaskedSpatialTemporalEncoder(
        dim=16,
        depth=2,
        heads=4,
        dropout=0.0,
        causal=False,
    ).eval()

    x = torch.randn(1, 4, 8, 16)
    sensor_mask = torch.tensor([[True, True, True, False]])
    time_mask = torch.tensor(
        [[True, True, True, True, True, False, False, False]]
    )
    valid = sensor_mask.unsqueeze(-1) & time_mask.unsqueeze(1)

    changed = x.clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 1000.0

    with torch.no_grad():
        output = encoder(
            x,
            sensor_mask=sensor_mask,
            time_mask=time_mask,
        )
        changed_output = encoder(
            changed,
            sensor_mask=sensor_mask,
            time_mask=time_mask,
        )

    assert torch.allclose(
        output[valid],
        changed_output[valid],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.count_nonzero(output[~valid]) == 0
    assert torch.count_nonzero(changed_output[~valid]) == 0


def test_all_invalid_attention_rows_remain_finite_and_zero():
    encoder = MaskedSpatialTemporalEncoder(
        dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
        causal=False,
    ).eval()
    x = torch.randn(2, 3, 5, 16)
    sensor_mask = torch.zeros(2, 3, dtype=torch.bool)
    time_mask = torch.zeros(2, 5, dtype=torch.bool)

    with torch.no_grad():
        output = encoder(
            x,
            sensor_mask=sensor_mask,
            time_mask=time_mask,
        )

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
