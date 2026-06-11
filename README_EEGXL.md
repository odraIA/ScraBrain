# EEG/MEG-XL Operations

This guide covers the EEG word-aligned MEG-XL evaluation services, sweeps,
monitor, comparison summaries, and artifact layout.

## Dataset Info

Generate or refresh dataset summaries:

```bash
python scripts/summarize_datasets_info.py datasets --output-dir datasets_info
```

Useful EEG download helpers already in this repo:

```bash
bash scripts/clone_openneuro_ds004408.sh
bash scripts/download_openneuro_eeg_docker.sh
bash scripts/export_openneuro_eeg_folder_tree_docker.sh
bash scripts/download_eegdash_docker.sh
```

`clone_openneuro_ds004408.sh` clones or updates the DataLad/Git-annex mirror
and materializes `stimuli/*.TextGrid` by default. Those TextGrid files are
required by the ds004408 word-aligned listening dataset.

Download the complete EEGDash NM000228 cache used by the reading evaluation:

```bash
docker compose build
bash scripts/download_eegdash_docker.sh
```

The downloader is based on `datasets_info/eegdash/testing.ipynb`: it queries
`NM000228` through the EEGDash API, then accesses each lazy `recording.raw` so
the raw BIDS files and sidecars are materialized under
`datasets/eegdash/data/nm000228`. For a quick server check before the full
download:

```bash
bash scripts/download_eegdash_docker.sh --limit 1
python scripts/smoke_eeg_word_datasets.py --eegdash-root datasets/eegdash/data/nm000228
```

## Smoke Tests

Run the lightweight EEG dataset smoke test without requiring a local EEGDash
checkout:

```bash
python scripts/smoke_eeg_word_datasets.py --skip-eegdash
```

Run all unittest smoke tests:

```bash
python -m unittest discover -s tests
```

Validate Docker services:

```bash
docker compose config
```

## Single Runs

Build the image:

```bash
docker compose build
```

Run EEG reading:

```bash
docker compose run --rm eval_eeg_reading
```

Run EEG listening:

```bash
docker compose run --rm eval_eeg_listening
```

Run pooled reading + listening:

```bash
docker compose run --rm eval_eeg_reading_listening
```

Run all three evaluations in order and start the EEG monitor:

```bash
bash run_eeg_evals_with_monitor.sh
```

Preview the same sequence without launching training containers:

```bash
bash run_eeg_evals_with_monitor.sh --dry-run
```

Common overrides:

```bash
EEG_TARGET_SFREQ=100 EEG_TOKENIZER_NAME=biocodec docker compose run --rm eval_eeg_reading
EEG_TRAIN_FROM_SCRATCH=true docker compose run --rm eval_eeg_listening
EEG_USE_PROMOTED_CHECKPOINT=true EEG_PROMOTED_CHECKPOINT=./checkpoints/eeg_promoted/stage_1_frequency_search/best_checkpoint.pt docker compose run --rm eval_eeg_reading_listening
```

## Independent Sweep

Preview candidates:

```bash
python scripts/make_eeg_sweep_plan.py --dry-run --limit 2
python scripts/run_eeg_sweep.py --dry-run --limit 2
```

Run through Docker:

```bash
EEG_SWEEP_FREQUENCIES=25,50 EEG_TOKENIZER_NAME=biocodec docker compose run --rm eeg_sweep
```

## Chained Sweep

Preview staged commands:

```bash
python scripts/run_eeg_chained_sweep.py --dry-run --limit 2
```

Run through Docker:

```bash
EEG_SELECTION_METRIC=balanced_top10_accuracy_retrieval250 EEG_SELECTION_MODE=max docker compose run --rm eeg_chained_sweep
```

Promotion records are written to `promotions/stage_N_name_promotion.json`.
The final lineage is written to `promotions/eeg_chained_sweep_lineage.json`.

## Monitor

Start the EEG-aware monitor:

```bash
docker compose up eeg_monitor
```

Open:

```text
http://localhost:8082
```

JSON endpoints:

```bash
curl http://localhost:8082/api/status
curl 'http://localhost:8082/api/log?exp=<experiment_name>'
curl http://localhost:8082/api/results
curl http://localhost:8082/api/compare
curl http://localhost:8082/api/chained_status
curl http://localhost:8082/api/eeg_eval_sequence
```

The existing MEG monitor remains available with:

```bash
docker compose up monitor
```

## Compare Results

Generate CSV and Markdown summaries:

```bash
python scripts/compare_eeg_results.py
```

Outputs:

```text
results/eeg_experiments_summary.csv
results/eeg_experiments_summary.md
results/chained_eeg_sweep_summary.csv
results/chained_eeg_sweep_summary.md
```

## Artifact Layout

Each run writes:

```text
logs/.../<experiment>/
  config_resolved.yaml
  stdout_stderr.log
  epoch_metrics.csv
  epoch_metrics.jsonl
  metrics_history.csv
  metrics_history.jsonl
  final_results.txt
  final_results.json
  run_metadata.json

checkpoints/.../<experiment>/
  checkpoint_latest.pt
  checkpoint_best.pt
  checkpoint_load_report.json

promotions/
  stage_N_name_promotion.json
  eeg_chained_sweep_lineage.json

.eeg_eval_sequence.json

wandb/
embeddings_cache/
hf_cache/
data/cache/
```

## Environment Variables

The EEG Docker services read:

```text
EEG_DATASETS_DIR
EEG_READING_ROOTS
EEG_LISTENING_ROOTS
EEG_TARGET_SFREQ
EEG_TOKENIZER_NAME
EEG_TRAIN_FROM_SCRATCH
EEG_USE_PROMOTED_CHECKPOINT
EEG_PROMOTED_CHECKPOINT
EEG_SWEEP_FREQUENCIES
EEG_SELECTION_METRIC
EEG_SELECTION_MODE
CRISS_CROSS_CHECKPOINT
BIOCODEC_CHECKPOINT
BRAINOMNI_CHECKPOINT
WANDB_MODE
WANDB_API_KEY
```

## Limitations

- BioCodec is the default functional tokenizer.
- BrainOmni and BrainTokenizer configs/checkpoints can be validated, but using
  them for encoding requires their external model implementation to be wired
  into `brainstorm/neuro_tokenizers/factory.py`.
- Reading datasets require usable fixation or word-onset metadata.
- Listening datasets require usable word or audio onset metadata.
- Tokenizer changes may require partial checkpoint loading because tokenizer
  codebook shape, quantizer count, or downsample ratio can differ.
- Full training is intentionally not part of the smoke test suite.
