# Continuous OpenNeuro EEG pre-training for ScraBrain

This bundle replaces word-aligned EEG segmentation **only for MEG-XL-style self-supervised pre-training**. It does not change the later word-classification pipeline.

## Why this implementation is needed

The original MEG pre-training path uses fixed-duration neural-signal windows:

```text
continuous MEG/EEG -> neural tokenizer -> temporal masking -> RVQ-token prediction
```

It does not require words, transcripts, T5 embeddings, or language-specific tokenization. The previous EEG loader instead required 50 word-aligned labels before creating one 150-second segment. That is why Japanese ds007808 produced zero segments.

## How ds007808 `listeningcovert` is handled

For every `task-listeningcovert` recording:

1. Read its matching `events.tsv`.
2. Select every row where `trial_type == listening`.
3. Use the complete interval `[onset, onset + duration]` by default.
4. Exclude every `covert` interval.
5. Concatenate all selected listening intervals from runs belonging to the same subject and session.
6. Split the virtual listening-only stream into fixed 150-second windows.
7. If a remainder exists, create a final overlapping window ending at the end of the stream. Therefore, no selected listening sample is discarded.
8. If an entire grouped stream is shorter than 150 seconds, repeat real listening samples cyclically rather than inserting covert data.

This means that 100% of the union of intervals labelled `listening` is represented in at least one training segment.

### Important methodological caveat

The listening intervals are not temporally adjacent in the original recording. Concatenating them introduces boundaries between trials and sometimes between runs. This avoids using covert EEG and preserves the existing fixed-length model interface, but those boundaries are artificial. The final remainder window can also overlap a previous window, so a small portion of listening data can appear twice.

A more invasive alternative would keep complete recordings and modify the model loss with a temporal validity mask. The present implementation deliberately avoids changing `CrissCrossTransformerModule` and its checkpoint compatibility.

## Files

### `brainstorm/data/openneuro_eeg_continuous_dataset.py`

Implements `OpenNeuroEEGContinuousDataset`.

It:

- discovers BIDS EEG files (`.edf`, `.bdf`, `.vhdr`);
- filters and resamples recordings with MNE;
- caches preprocessed recordings as HDF5;
- obtains EEG sensor positions and marks sensor type as EEG (`2`);
- creates full-recording streams for `task-listening`;
- creates listening-only virtual streams for `task-listeningcovert`;
- returns the signal under the existing `meg` key, preserving model compatibility;
- keeps `segment_index` and `recordings`, so `RecordingShuffleSampler` still works.

### `brainstorm/data/eeg_continuous_multi_datamodule.py`

Implements a continuous version of `MultiEEGDataModule` with the same batch format as the existing trainer:

```python
(eeg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids)
```

The old word-alignment arguments remain accepted but are ignored, so the existing training entrypoint can call this DataModule without changing its argument list.

### `brainstorm/train_criss_cross_eeg_continuous.py`

Small compatibility entrypoint. It reuses the complete existing training script and swaps only its DataModule class. This avoids maintaining a duplicate copy of the trainer.

### `configs/train_criss_cross_eeg_multi_continuous.yaml`

Hydra configuration inheriting the existing model/training settings. It enables:

```yaml
listeningcovert_policy: listening_only
listening_trial_type: listening
listening_interval_start: onset
group_listeningcovert_by: subject_session
cover_all_samples: true
short_stream_policy: repeat
```

Change `listening_interval_start` to `wav_onset` only when you want to exclude the short interval between event onset and actual audio onset.

### `scripts/check_ds007808_listening_coverage.py`

Fast metadata-only check. It does not read or preprocess EDF files. It reports all listening/covert rows and the total union of listening intervals.

### `scripts/smoke_eeg_continuous_dataset.py`

End-to-end smoke test on one or a few sessions. It reads EEG, creates the cache, builds listening-only windows, loads one sample, and verifies that no words are used.

### `scripts/test_continuous_segmentation.py`

Small test for interval merging and final-window coverage.


## Installation

Extract this bundle into the root of `ScraBrain`:

```bash
cd ~/proyectos/meegxl/ScraBrain
unzip -o continuous_eeg_implementation.zip
```

No new Python dependency is required.

## Checks before training

### 1. Count all listening intervals selected from `listeningcovert`

```bash
cd ~/proyectos/meegxl/ScraBrain

docker compose run --rm --no-deps -T eeg_sweep \
  bash -lc 'uv run --no-sync python scripts/check_ds007808_listening_coverage.py \
    --data-root /workspace/datasets/OpenNeuroEEG_ds007808'
```

To inspect only the current validation sessions:

```bash
docker compose run --rm --no-deps -T eeg_sweep \
  bash -lc 'uv run --no-sync python scripts/check_ds007808_listening_coverage.py \
    --data-root /workspace/datasets/OpenNeuroEEG_ds007808 \
    --sessions \
      ses-20240621 ses-20240624 ses-20240625 \
      ses-20240626 ses-20240627 ses-20240628 \
      ses-20240701 ses-20241119 ses-20241227'
```

### 2. Test the segmentation helpers

```bash
docker compose run --rm --no-deps eeg_sweep \
  bash -lc 'uv run --no-sync python scripts/test_continuous_segmentation.py'
```

### 3. End-to-end smoke test on one session

This first run preprocesses and caches the selected EDF files, so it is slower than the metadata check.

```bash
docker compose run --rm --no-deps eeg_sweep \
  bash -lc 'uv run --no-sync python scripts/smoke_eeg_continuous_dataset.py \
    --data-root /workspace/datasets/OpenNeuroEEG_ds007808 \
    --sessions ses-20240624 \
    --tasks listeningcovert \
    --segment-length 30 \
    --target-sfreq 50 \
    --l-freq 0.1 \
    --h-freq 24'
```

Expected properties in the JSON output:

```text
content_mode = listening_only
no_word_alignment_used = true
covert_intervals_excluded_by_trial_type_filter = true
segments > 0
```

## Start one 50 Hz training

```bash
cd ~/proyectos/meegxl/ScraBrain

docker compose run --rm --no-deps eeg_sweep \
  bash -lc 'uv run --no-sync python -m \
    brainstorm.train_criss_cross_eeg_continuous \
    --config-name=train_criss_cross_eeg_multi_continuous'
```

For the beta-band experiment currently used in your commands:

```bash
docker compose run --rm --no-deps eeg_sweep \
  bash -lc 'uv run --no-sync python -m \
    brainstorm.train_criss_cross_eeg_continuous \
    --config-name=train_criss_cross_eeg_multi_continuous \
    data.target_sfreq=50.0 \
    model.sampling_rate=50 \
    data.l_freq=13.0 \
    data.h_freq=24.0 \
    data.cache_dir=./data/cache/eeg_continuous/beta_13_24_biocodec \
    logging.experiment_name=eeg_continuous_beta_13_24_biocodec_scratch_seed42'
```

## Cache behavior

Caches are written below:

```text
data/cache/eeg_continuous/<dataset_name>/continuous/
```

The cache filename depends on:

- raw file path, size, and modification time;
- low/high filter frequencies;
- target sampling rate;
- channel filter.

Changing frequency settings therefore creates a separate compatible cache.

## What this does not do

- It does not tokenize Japanese.
- It does not use the `value` transcript column.
- It does not perform word classification.
- It does not include `trial_type=covert` in the listening-only stream.
- It does not change BioCodec, BrainOmni, the transformer, or the pre-training loss.
