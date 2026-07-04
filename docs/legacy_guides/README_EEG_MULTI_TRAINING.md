# EEG multi-dataset training files

Copy these files into the repository root preserving paths:

```text
brainstorm/train_criss_cross_eeg_multi.py
configs/train_criss_cross_eeg_multi_feq.yaml
docker-compose.eeg-train.yml
scripts/run_eeg_multi_training_sweep.sh
```

The training script is based on `brainstorm/train_criss_cross_multi.py`, but it imports `MultiEEGDataModule` and writes local artifacts for every run.

## Single training run

```bash
python -m brainstorm.train_criss_cross_eeg_multi \
  --config-name train_criss_cross_eeg_multi_feq \
  data.target_sfreq=128.0 \
  model.sampling_rate=128 \
  data.l_freq=30.0 \
  data.h_freq=55.0 \
  logging.experiment_name=eeg_multi_gamma_30_55_biocodec
```

## Docker single run

```bash
docker compose -f docker-compose.yml -f docker-compose.eeg-train.yml build eeg_train_multi

docker compose -f docker-compose.yml -f docker-compose.eeg-train.yml run --rm eeg_train_multi \
  bash -lc 'uv run --no-sync python -m brainstorm.train_criss_cross_eeg_multi \
    --config-name train_criss_cross_eeg_multi_feq \
    data.target_sfreq=128.0 model.sampling_rate=128 \
    data.l_freq=30.0 data.h_freq=55.0 \
    logging.experiment_name=eeg_multi_gamma_30_55_biocodec'
```

## Full sequential sweep

```bash
bash scripts/run_eeg_multi_training_sweep.sh
```

Useful quick test:

```bash
EEG_SWEEP_LIMIT=1 EEG_MAX_STEPS=20 WANDB_MODE=offline bash scripts/run_eeg_multi_training_sweep.sh
```

Run only some tokenizers/bands:

```bash
EEG_TOKENIZERS="biocodec brainomni_base" \
EEG_BANDS="beta_13_24 gamma_30_55 high_gamma_70_120" \
WANDB_MODE=offline \
bash scripts/run_eeg_multi_training_sweep.sh
```

Pretrained initialization instead of scratch:

```bash
EEG_INIT_MODE=pretrained \
CRISS_CROSS_CHECKPOINT=./checkpoints/baseline/meg-xl-med.ckpt \
bash scripts/run_eeg_multi_training_sweep.sh
```

## Artifacts

Each run writes:

```text
logs/eeg_multi_training/<experiment>/config_resolved.yaml
logs/eeg_multi_training/<experiment>/stdout_stderr.log
logs/eeg_multi_training/<experiment>/epoch_metrics.csv
logs/eeg_multi_training/<experiment>/epoch_metrics.jsonl
logs/eeg_multi_training/<experiment>/final_results.txt
logs/eeg_multi_training/<experiment>/final_results.json
checkpoints/eeg_multi_training/<experiment>/checkpoint_best.pt
checkpoints/eeg_multi_training/<experiment>/checkpoint_latest.pt
```

The sweep writes a global summary at:

```text
results/eeg_multi_training_sweep/<timestamp>/final_results.txt
results/eeg_multi_training_sweep/<timestamp>/runs.tsv
```

## Important tokenizer note

The YAML and launcher pass tokenizer names and checkpoint paths, but the repo must actually support those names inside `brainstorm/neuro_tokenizers/factory.py`. If `brainomni_base`, `brainomni_tiny`, or `braintokenizer` are not implemented in the factory, those runs will fail and the sweep summary will mark them as failed.
