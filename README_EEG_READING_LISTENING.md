# Joint EEG reading + listening pre-training

This is the third continuous EEG training variant for ScraBrain. It pre-trains the existing MEG-XL Criss-Cross Transformer on both speech listening and natural reading EEG without changing the model architecture or the masked-token objective.

## Included datasets

| Domain | Dataset type | Configured task | Expected root inside Docker |
|---|---|---|---|
| Listening | OpenNeuro ds004408 | `listening` | `/workspace/datasets/OpenNeuroEEG_ds004408` |
| Listening | OpenNeuro ds007808 | `listening`, `listeningcovert` | `/workspace/datasets/OpenNeuroEEG_ds007808` |
| Listening | SparrKULee | `listeningActive` | `/workspace/datasets/sparrkulee` |
| Reading | EEGDash NM000228 | `delong`, `control` | `/workspace/datasets/eegdash/data/nm000228` |
| Reading | ZuCo 2.0 | `NR` only | `/workspace/datasets/zuco2/data/zuco2` |

The configuration is:

```text
configs/train_criss_cross_eeg_reading_listening_continuous.yaml
```

## Training semantics

- Input context: complete 150-second windows.
- Windowing: non-overlapping windows inside one physical run.
- Incomplete final remainders: discarded, never repeated or zero-padded.
- Preprocessing order: band-pass at the original sampling rate, followed by MNE resampling.
- Default sampling rate: 50 Hz.
- Default tokenizer: BioCodec.
- Objective: masked RVQ-token prediction.
- EEG sensor type: `2`; electrode positions are retained and MEG coil orientations are zeroed.
- No word labels, transcripts, T5 embeddings or word alignment are required for pre-training.

For ds007808 `listeningcovert`, the complete run remains available as context, but only intervals labelled `trial_type=listening` can be selected as reconstruction targets. The other four dataset types use their complete selected recordings as valid targets.

## Validation splits

The existing explicit holdouts remain unchanged for OpenNeuro and SparrKULee.

EEGDash and ZuCo use deterministic subject-held-out splits because their local releases do not share the same directory/split convention:

```yaml
split_axis: subject
val_fraction: 0.10  # EEGDash
split_seed: 42
```

```yaml
split_axis: subject
val_fraction: 0.15  # ZuCo
split_seed: 42
```

The selected validation subject IDs are printed at startup and remain stable for the same files, fraction and seed.

## Prepare the data

### OpenNeuro

The raw EEG files must be materialized, not unresolved git-annex links.

Expected directories:

```text
datasets/OpenNeuroEEG_ds004408/
datasets/OpenNeuroEEG_ds007808/
```

### SparrKULee

Expected directory:

```text
datasets/sparrkulee/
```

The loader accepts the materialized BIDS `.bdf` files and the project-specific `.bdf.gz` workflow already used by ScraBrain.

### EEGDash NM000228

Download both configured tasks through Docker:

```bash
bash scripts/download_eegdash_docker.sh \
  --task delong \
  --task control
```

The resulting training root should be:

```text
datasets/eegdash/data/nm000228/
```

The continuous loader currently consumes materialized BIDS EEG files supported by the shared MNE path (`.bdf`, `.edf` and BrainVision `.vhdr`). CNT recordings requiring the dataset-specific repaired parser are not included by this loader.

### ZuCo 2.0

Only Natural Reading (`task1 - NR`) is used. The required files are the preprocessed MATLAB v7.3/HDF5 EEG files:

```text
datasets/zuco2/data/zuco2/task1 - NR/Preprocessed/<SUBJECT>/*_NR*_EEG.mat
```

Task-specific reading/annotation (`TSR`) is intentionally excluded from this training variant.

## Build the Docker image

From the repository root:

```bash
docker compose \
  -f docker-compose.eeg-reading-listening.yml \
  build eeg_train_reading_listening
```

## Run the default training

```bash
bash scripts/run_eeg_reading_listening_training.sh
```

The launcher accepts normal Hydra overrides. For example, to use two GPUs with FSDP:

```bash
EEG_GPU=0,1 EEG_GPU_COUNT=2 \
  bash scripts/run_eeg_reading_listening_training.sh \
  trainer.devices=2 \
  trainer.strategy=fsdp \
  model.use_gradient_checkpointing=true
```

To initialize from the MEG-XL checkpoint instead of training from scratch:

```bash
bash scripts/run_eeg_reading_listening_training.sh \
  model.train_from_scratch=false \
  model.criss_cross_checkpoint=./checkpoints/baseline/meg-xl-med.ckpt \
  logging.experiment_name=criss-cross-eeg-reading-listening-50hz-biocodec-megxl-init-seed42
```

## Run the integration tests

```bash
docker compose \
  -f docker-compose.eeg-reading-listening.yml \
  run --rm --no-deps eeg_train_reading_listening \
  bash -lc 'uv run --no-sync pytest -q \
    tests/test_eeg_reading_listening_datamodule.py \
    tests/test_eeg_continuity_masks.py'
```

## Outputs

The default run writes to:

```text
data/cache/eeg_reading_listening_continuous/
logs/eeg_reading_listening_training/
checkpoints/eeg_reading_listening_training/
wandb/
```

The existing local CSV logger, metrics files, final-results metadata, checkpoints and optional Weights & Biases logging remain active.
