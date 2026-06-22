# Staged EEG reading → listening sweep

This launcher runs one independent experiment per GPU and automatically assigns the next queued experiment when a GPU becomes free. Every experiment has two consecutive stages:

1. train on continuous **reading** EEG (EEGDash NM000228 + ZuCo NR);
2. train on continuous **listening** EEG (ds004408 + ds007808 + SparrKULee), initialized from the best checkpoint produced in stage 1.

The existing Docker working directory, dataset mounts and preprocessing cache are preserved. In particular, all stages keep using:

```text
/workspace/datasets/...
./data/cache/eeg_reading_listening_continuous
./logs/eeg_reading_listening_training
./checkpoints/eeg_reading_listening_training
```

## Experiment count

The fixed-50-Hz control matrix contains:

```text
6 bands × 4 tokenizers × 2 initializations = 48 pipelines
```

Four bands exceed the 25 Hz Nyquist limit of a 50 Hz output signal, so they are repeated with a higher target sampling rate:

```text
4 affected bands × 4 tokenizers × 2 initializations = 32 additional pipelines
```

Total:

```text
80 experiment pipelines
160 training stages (80 reading + 80 listening)
80 final listening models
```

Frequency definitions:

| Band | Range | Fixed control | Nyquist-aware repetition |
|---|---:|---:|---:|
| alpha | 8–12 Hz | 50 Hz | — |
| beta | 13–24 Hz | 50 Hz | — |
| beta-gamma | 13–45 Hz | 50 Hz | 100 Hz |
| low-gamma | 30–45 Hz | 50 Hz | 100 Hz |
| gamma | 30–55 Hz | 50 Hz | 128 Hz |
| high-gamma | 70–120 Hz | 50 Hz | 250 Hz |

The fixed-50-Hz runs above 25 Hz are deliberate controls of the current filter-then-resample pipeline. MNE anti-alias filtering means those runs do **not** preserve the complete requested gamma band after resampling; the Nyquist-aware repetitions are the scientifically valid versions for retaining those frequencies.

## Run

From the repository root:

```bash
bash scripts/run_eeg_reading_then_listening_sweep.sh
```

Defaults:

- GPU workers: `0 1`;
- one training process per GPU;
- initial batch-size candidates: `16 12 8 6 4 2 1`;
- initializations: `scratch pretrained`;
- tokenizers: `biocodec brainomni_base brainomni_tiny braintokenizer`;
- 50 epochs per stage, inherited from the existing config;
- continue with the next queued pipeline after a failure.

## Automatic batch sizing

Before the first full run for a `(stage, band, sampling profile, tokenizer, GPU-memory size)` combination, the launcher executes a two-step probe from the largest candidate to the smallest. The first candidate that completes is cached in `batch_sizes.tsv`. If a real run still raises an OOM, it is automatically retried with the next smaller candidate.

To start conservatively at batch size 4 and only try smaller values:

```bash
EEG_BATCH_CANDIDATES="4 2 1" \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

To disable probing and force batch size 4:

```bash
EEG_AUTO_BATCH=false EEG_DEFAULT_BATCH_SIZE=4 \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

## Useful smoke tests

Generate the full queue without launching Docker:

```bash
EEG_DRY_RUN=1 bash scripts/run_eeg_reading_then_listening_sweep.sh
```

Run only two pipelines, with ten steps per stage:

```bash
EEG_SWEEP_LIMIT=2 EEG_MAX_STEPS=10 EEG_BATCH_CANDIDATES="4 2 1" \
  bash scripts/run_eeg_reading_then_listening_sweep.sh
```

Use different GPU IDs:

```bash
EEG_GPUS="2 3" bash scripts/run_eeg_reading_then_listening_sweep.sh
```

## Results

Each launch creates:

```text
results/eeg_reading_listening_sweep/<timestamp>/
├── jobs.tsv
├── runs.tsv
├── batch_sizes.tsv
├── sweep_metadata.txt
├── final_results.txt
├── batch_probes/
└── stages/
```

The actual model outputs remain in the established locations:

```text
logs/eeg_reading_listening_training/<experiment>/
checkpoints/eeg_reading_listening_training/<experiment>/
```

A completed reading stage exposes `checkpoint_best.pt` (or `checkpoint_latest.pt` as fallback), which is passed to the corresponding listening stage through `model.promoted_checkpoint`.
