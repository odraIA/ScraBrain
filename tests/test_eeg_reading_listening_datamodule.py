from pathlib import Path

from brainstorm.data.eeg_continuous_masked_dataset import (
    ContinuityAwareEEGDashDataset,
    ContinuityAwareEEGMixin,
    ContinuityAwareZuCoEEGDataset,
)
from brainstorm.data.eeg_continuous_multi_datamodule import MultiEEGDataModule
from brainstorm.data.eegdash_eeg_continuous_dataset import EEGDashEEGContinuousDataset
from brainstorm.data.zuco_eeg_continuous_dataset import ZuCoEEGContinuousDataset


def make_datamodule():
    return MultiEEGDataModule(
        datasets_config=[],
        num_workers=0,
        persistent_workers=False,
        infer_max_channel_dim=False,
    )


def test_reading_dataset_aliases_and_default_tasks():
    datamodule = make_datamodule()

    assert datamodule._canonical_type("NM000228") == "eegdash"
    assert datamodule._canonical_type("zuco2") == "zuco"
    assert datamodule._default_tasks("eegdash") == ["delong", "control"]
    assert datamodule._default_tasks("zuco") == ["NR"]


def test_eegdash_fraction_split_uses_nested_materialized_subjects(tmp_path: Path):
    root = tmp_path / "eegdash" / "data" / "nm000228"
    subjects = [
        "sub-birm0001",
        "sub-bris0001",
        "sub-edin0001",
        "sub-glas0001",
        "sub-kent0001",
        "sub-lond0001",
        "sub-oxfo0001",
        "sub-york0001",
    ]
    for subject in subjects:
        (root / subject / "eeg").mkdir(parents=True)

    config = {
        "type": "eegdash",
        "data_root": str(tmp_path / "eegdash" / "data"),
        "subjects": None,
        "sessions": None,
        "val_fraction": 0.25,
        "split_axis": "subject",
        "split_seed": 7,
    }
    datamodule = make_datamodule()

    train_subjects, train_sessions = datamodule._split_filters(config, "train")
    val_subjects, val_sessions = datamodule._split_filters(config, "val")

    assert train_sessions is None
    assert val_sessions is None
    assert len(val_subjects) == 2
    assert set(train_subjects).isdisjoint(val_subjects)
    assert set(train_subjects) | set(val_subjects) == set(subjects)
    assert datamodule._split_filters(config, "val")[0] == val_subjects


def test_zuco_fraction_split_discovers_preprocessed_subject_directories(tmp_path: Path):
    root = tmp_path / "zuco2" / "data" / "zuco2" / "task1 - NR" / "Preprocessed"
    subjects = ["YAC", "YAG", "YAK", "YDG", "YDR", "YFR"]
    for subject in subjects:
        (root / subject).mkdir(parents=True)

    config = {
        "type": "zuco",
        "data_root": str(tmp_path / "zuco2"),
        "subjects": None,
        "sessions": None,
        "val_fraction": 0.34,
        "split_axis": "subject",
        "split_seed": 42,
    }
    datamodule = make_datamodule()

    train_subjects, _ = datamodule._split_filters(config, "train")
    val_subjects, _ = datamodule._split_filters(config, "val")

    expected = {f"sub-{subject}" for subject in subjects}
    assert len(val_subjects) == 2
    assert set(train_subjects).isdisjoint(val_subjects)
    assert set(train_subjects) | set(val_subjects) == expected


def test_reading_datasets_use_continuity_aware_mixin():
    assert issubclass(ContinuityAwareEEGDashDataset, ContinuityAwareEEGMixin)
    assert issubclass(ContinuityAwareEEGDashDataset, EEGDashEEGContinuousDataset)
    assert issubclass(ContinuityAwareZuCoEEGDataset, ContinuityAwareEEGMixin)
    assert issubclass(ContinuityAwareZuCoEEGDataset, ZuCoEEGContinuousDataset)
