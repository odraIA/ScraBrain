import torch

from megxl_adapters.collate import megxl_collate
from megxl_adapters.sensor_mask import apply_sensor_mask, pad_channels


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}")


def test_pad_channels_with_padding():
    x = torch.arange(6, dtype=torch.float32).view(2, 3)

    x_pad, sensor_mask = pad_channels(x, max_channels=4)

    assert x_pad.shape == (4, 3)
    assert sensor_mask.dtype == torch.bool
    assert sensor_mask.tolist() == [True, True, False, False]
    assert torch.equal(x_pad[:2], x)
    assert torch.equal(x_pad[2:], torch.zeros(2, 3))


def test_pad_channels_exact_channels():
    x = torch.ones(3, 5)

    x_pad, sensor_mask = pad_channels(x, max_channels=3)

    assert x_pad.shape == (3, 5)
    assert sensor_mask.tolist() == [True, True, True]
    assert torch.equal(x_pad, x)


def test_pad_channels_too_many_channels():
    x = torch.zeros(5, 3)

    _assert_raises(ValueError, lambda: pad_channels(x, max_channels=4))


def test_apply_sensor_mask_raw_batch():
    x = torch.ones(2, 4, 3)
    sensor_mask = torch.tensor(
        [[True, True, False, False], [True, False, True, False]]
    )

    masked = apply_sensor_mask(x, sensor_mask)

    assert masked.shape == x.shape
    assert torch.equal(masked[0, 2:], torch.zeros(2, 3))
    assert torch.equal(masked[1, 1], torch.zeros(3))
    assert torch.equal(masked[1, 3], torch.zeros(3))
    assert torch.equal(masked[1, 0], torch.ones(3))
    assert torch.equal(masked[1, 2], torch.ones(3))


def test_apply_sensor_mask_scalogram_batch():
    x = torch.ones(2, 4, 5, 3)
    sensor_mask = torch.tensor(
        [[True, False, True, False], [False, True, True, False]]
    )

    masked = apply_sensor_mask(x, sensor_mask)

    assert masked.shape == x.shape
    assert torch.equal(masked[0, 1], torch.zeros(5, 3))
    assert torch.equal(masked[0, 3], torch.zeros(5, 3))
    assert torch.equal(masked[1, 0], torch.zeros(5, 3))
    assert torch.equal(masked[1, 3], torch.zeros(5, 3))
    assert torch.equal(masked[0, 0], torch.ones(5, 3))
    assert torch.equal(masked[1, 2], torch.ones(5, 3))


def test_megxl_collate_shapes():
    batch = [
        {
            "meg": torch.ones(2, 4),
            "label": 1,
            "dataset_id": "a",
            "subject_id": "s0",
        },
        {
            "meg": torch.ones(3, 4) * 2,
            "label": 2,
            "dataset_id": "b",
            "subject_id": "s1",
        },
    ]

    output = megxl_collate(batch, max_channels=4)

    assert output["meg"].shape == (2, 4, 4)
    assert output["sensor_mask"].shape == (2, 4)
    assert output["sensor_mask"].dtype == torch.bool
    assert output["label"].shape == (2,)
    assert output["label"].dtype == torch.long
    assert output["dataset_id"] == ["a", "b"]
    assert output["subject_id"] == ["s0", "s1"]
    assert output["sensor_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, False],
    ]
    assert torch.equal(output["meg"][0, 2:], torch.zeros(2, 4))


if __name__ == "__main__":
    for test in [
        test_pad_channels_with_padding,
        test_pad_channels_exact_channels,
        test_pad_channels_too_many_channels,
        test_apply_sensor_mask_raw_batch,
        test_apply_sensor_mask_scalogram_batch,
        test_megxl_collate_shapes,
    ]:
        test()
    print("ok")
