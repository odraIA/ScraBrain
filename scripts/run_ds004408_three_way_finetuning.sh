#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare ds004408 as word-aligned EEG and run the same three-way comparison
# used for Weissbart and Alice: random init, EEG-from-scratch checkpoint, and
# EEG initialized from MEG-XL. Runs are sequential on one GPU and survive SSH.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_ds004408_three_way_finetuning.sh"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

DS004408_ROOT="${DS004408_ROOT:-./datasets/OpenNeuroEEG_ds004408}"
BIOCODEC_CHECKPOINT="${BIOCODEC_CHECKPOINT:-./brainstorm/neuro_tokenizers/biocodec_ckpt.pt}"
MEGXL_ARCH_CHECKPOINT="${MEGXL_ARCH_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
SCRATCH_EEG_CHECKPOINT="${SCRATCH_EEG_CHECKPOINT:-./checkpoints/eeg_full_band_reading_then_listening_compare/20260624_182700/eeg_full_band_0p1_50_fixed50_50hz_biocodec_from_scratch_listening_seed42/checkpoint_best.pt}"
PRETRAINED_EEG_CHECKPOINT="${PRETRAINED_EEG_CHECKPOINT:-./checkpoints/eeg_full_band_reading_then_listening_compare/20260624_182700/eeg_full_band_0p1_50_fixed50_50hz_biocodec_pretrained_listening_seed42/checkpoint_best.pt}"

TRAIN_PCT="${TRAIN_PCT:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREPARE_WORD_ALIGNED="${PREPARE_WORD_ALIGNED:-1}"
WARM_WORD_ALIGNED_CACHE="${WARM_WORD_ALIGNED_CACHE:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-./results/ds004408_three_way/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-./logs/word_classification_ds004408_eeg/$RUN_ID}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/word_classification_ds004408_eeg/$RUN_ID}"
HYDRA_ROOT="${HYDRA_ROOT:-./logs/hydra/ds004408_three_way/$RUN_ID}"
WORD_ALIGNED_OUTPUT="${WORD_ALIGNED_OUTPUT:-$RESULTS_ROOT/word_aligned}"
WORD_ALIGNED_CACHE="${WORD_ALIGNED_CACHE:-./data/cache/ds004408_word_aligned_v2}"
MASTER_LOG="${MASTER_LOG:-$ROOT_DIR/logs/ds004408_three_way_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/ds004408_three_way_${RUN_ID}.pid}"
CURRENT_CONTAINER_FILE="$RESULTS_ROOT/current_container.txt"
STATUS_FILE="$RESULTS_ROOT/runs.tsv"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$HYDRA_ROOT" \
  "$WORD_ALIGNED_OUTPUT" "$(dirname "$MASTER_LOG")"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: Missing $2: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "ERROR: Missing $2: $1" >&2; exit 2; }
}

preflight() {
  command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not available" >&2; exit 2; }
  require_dir "$DS004408_ROOT" "ds004408 dataset directory"
  compgen -G "$DS004408_ROOT/sub-*/eeg/*_eeg.vhdr" >/dev/null || {
    echo "ERROR: No ds004408 BrainVision files under $DS004408_ROOT" >&2; exit 2;
  }
  compgen -G "$DS004408_ROOT/stimuli/*.TextGrid" >/dev/null || {
    echo "ERROR: No materialized TextGrids. Run scripts/clone_openneuro_ds004408.sh" >&2; exit 2;
  }
  require_file "$BIOCODEC_CHECKPOINT" "BioCodec checkpoint"
  require_file "$MEGXL_ARCH_CHECKPOINT" "MEG-XL architecture checkpoint"
  require_file "$SCRATCH_EEG_CHECKPOINT" "EEG-from-scratch checkpoint"
  require_file "$PRETRAINED_EEG_CHECKPOINT" "MEG-XL-initialized EEG checkpoint"
}
preflight

if [[ "${DS004408_THREE_WAY_WORKER:-0}" != "1" ]]; then
  nohup env DS004408_THREE_WAY_WORKER=1 RUN_ID="$RUN_ID" MASTER_LOG="$MASTER_LOG" \
    PID_FILE="$PID_FILE" GPU="$GPU" WANDB_MODE="$WANDB_MODE" BUILD_IMAGE="$BUILD_IMAGE" \
    DS004408_ROOT="$DS004408_ROOT" BIOCODEC_CHECKPOINT="$BIOCODEC_CHECKPOINT" \
    MEGXL_ARCH_CHECKPOINT="$MEGXL_ARCH_CHECKPOINT" \
    SCRATCH_EEG_CHECKPOINT="$SCRATCH_EEG_CHECKPOINT" \
    PRETRAINED_EEG_CHECKPOINT="$PRETRAINED_EEG_CHECKPOINT" TRAIN_PCT="$TRAIN_PCT" \
    NUM_EPOCHS="$NUM_EPOCHS" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    PREPARE_WORD_ALIGNED="$PREPARE_WORD_ALIGNED" \
    WARM_WORD_ALIGNED_CACHE="$WARM_WORD_ALIGNED_CACHE" RESULTS_ROOT="$RESULTS_ROOT" \
    LOG_ROOT="$LOG_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" HYDRA_ROOT="$HYDRA_ROOT" \
    WORD_ALIGNED_OUTPUT="$WORD_ALIGNED_OUTPUT" WORD_ALIGNED_CACHE="$WORD_ALIGNED_CACHE" \
    bash "$SCRIPT_PATH" >> "$MASTER_LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  echo "$RUN_ID" > "$ROOT_DIR/ds004408_three_way.latest"
  echo "ds004408 pipeline launched. PID: $(cat "$PID_FILE")"
  echo "Log: $MASTER_LOG"
  echo "Results: $RESULTS_ROOT"
  exit 0
fi

echo $$ > "$PID_FILE"
printf 'order\tlabel\tcontainer\tstatus\texit_code\tstarted_at\tfinished_at\n' > "$STATUS_FILE"

if [[ "$BUILD_IMAGE" == "1" || "$BUILD_IMAGE" == "true" ]]; then
  docker compose build eval_eeg_listening
fi

CURRENT_EXPERIMENT=""
CURRENT_CONTAINER=""
trap 'status=$?; echo "ERROR: ${CURRENT_EXPERIMENT:-unknown} failed ($status)" >&2; [[ -z "$CURRENT_CONTAINER" ]] || echo "Inspect: docker logs --tail 200 $CURRENT_CONTAINER" >&2; exit $status' ERR

append_status() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$STATUS_FILE"
}

prepare_word_aligned() {
  [[ "$PREPARE_WORD_ALIGNED" == "1" || "$PREPARE_WORD_ALIGNED" == "true" ]] || return 0
  local started finished warm_flag="--no-warm-cache"
  started="$(date --iso-8601=seconds)"
  [[ "$WARM_WORD_ALIGNED_CACHE" == "1" || "$WARM_WORD_ALIGNED_CACHE" == "true" ]] && warm_flag="--warm-cache"
  CURRENT_EXPERIMENT="prepare_word_aligned"

  env EEG_GPU="$GPU" WANDB_MODE="$WANDB_MODE" \
    docker compose run --rm --no-deps -e "NVIDIA_VISIBLE_DEVICES=$GPU" \
      -e "WANDB_MODE=$WANDB_MODE" eval_eeg_listening \
      uv run --no-sync python scripts/prepare_ds004408_word_aligned.py \
        --root "$DS004408_ROOT" --cache-dir "$WORD_ALIGNED_CACHE" \
        --output-dir "$WORD_ALIGNED_OUTPUT" --eeg-sensor-type grad \
        --montage-name biosemi128 --drop-bad-channels "$warm_flag"

  require_file "$WORD_ALIGNED_OUTPUT/summary.json" "word-aligned summary"
  require_file "$WORD_ALIGNED_OUTPUT/word_aligned_manifest.csv" "word-aligned manifest"
  require_file "$WORD_ALIGNED_OUTPUT/alignment_report.json" "alignment report"
  finished="$(date --iso-8601=seconds)"
  append_status 0 prepare_word_aligned docker-compose-run-rm COMPLETED 0 "$started" "$finished"
}

run_experiment() {
  local order="$1" label="$2" train_from_scratch="$3" init_checkpoint="$4" compare="$5"
  local experiment="ds004408_${label}_${RUN_ID}" container="ds004408_${label}_${RUN_ID}"
  local save_dir="$LOG_ROOT/$label" checkpoint_dir="$CHECKPOINT_ROOT/$label"
  local hydra_dir="$HYDRA_ROOT/$label" started finished exit_code
  local extra_env=()

  CURRENT_EXPERIMENT="$experiment"; CURRENT_CONTAINER="$container"
  mkdir -p "$save_dir" "$checkpoint_dir" "$hydra_dir"
  if docker container inspect "$container" >/dev/null 2>&1; then
    [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]] || {
      echo "ERROR: container already running: $container" >&2; exit 3;
    }
    docker rm "$container" >/dev/null
  fi
  if [[ "$compare" == "1" ]]; then
    extra_env+=(
      -e "MEGXL_COMPARISON_RUNS=random_init=$LOG_ROOT/random_init;eeg_from_scratch=$LOG_ROOT/eeg_from_scratch;eeg_pretrained=$LOG_ROOT/eeg_pretrained"
      -e "MEGXL_COMPARISON_OUTPUT=$RESULTS_ROOT"
    )
  fi

  started="$(date --iso-8601=seconds)"; echo "$container" > "$CURRENT_CONTAINER_FILE"
  echo "START $experiment | GPU=$GPU | checkpoint=$init_checkpoint"

  env EEG_GPU="$GPU" WANDB_MODE="$WANDB_MODE" \
    docker compose run -d --no-deps --name "$container" \
      -e "NVIDIA_VISIBLE_DEVICES=$GPU" -e "WANDB_MODE=$WANDB_MODE" \
      "${extra_env[@]}" eval_eeg_listening \
      uv run --no-sync python -m scripts.evaluate_ds004408_word_classification \
        --config-name=ds004408_word_finetuning \
        "model.train_from_scratch=$train_from_scratch" model.use_promoted_checkpoint=false \
        model.promoted_checkpoint=null "model.criss_cross_checkpoint=$init_checkpoint" \
        model.tokenizer_name=biocodec "model.tokenizer_checkpoint=$BIOCODEC_CHECKPOINT" \
        "data.root=$DS004408_ROOT" "data.cache_dir=$WORD_ALIGNED_CACHE" \
        "data.train_pct=$TRAIN_PCT" "training.num_epochs=$NUM_EPOCHS" \
        "training.batch_size=$BATCH_SIZE" "training.num_workers=$NUM_WORKERS" \
        'evaluation.retrieval_set_sizes=[50,250]' evaluation.k=10 \
        "logging.experiment_name=$experiment" "logging.save_dir=$save_dir" \
        "logging.checkpoint_dir=$checkpoint_dir" "hydra.run.dir=$hydra_dir"

  docker logs --follow "$container" || true
  exit_code="$(docker wait "$container")"; finished="$(date --iso-8601=seconds)"
  if [[ "$exit_code" != "0" ]]; then
    append_status "$order" "$label" "$container" FAILED "$exit_code" "$started" "$finished"
    echo "ERROR: $container exited with $exit_code" >&2; exit "$exit_code"
  fi

  require_file "$save_dir/final_results.json" "final results"
  require_file "$save_dir/paper_test_metrics.csv" "paper metrics"
  require_file "$save_dir/paper_report_manifest.json" "report manifest"
  require_file "$checkpoint_dir/checkpoint_best.pt" "best checkpoint"
  append_status "$order" "$label" "$container" COMPLETED "$exit_code" "$started" "$finished"
  docker rm "$container" >/dev/null; CURRENT_CONTAINER=""
}

prepare_word_aligned
run_experiment 1 random_init true "$MEGXL_ARCH_CHECKPOINT" 0
run_experiment 2 eeg_from_scratch false "$SCRATCH_EEG_CHECKPOINT" 0
run_experiment 3 eeg_pretrained false "$PRETRAINED_EEG_CHECKPOINT" 1

CURRENT_EXPERIMENT="summary"; rm -f "$CURRENT_CONTAINER_FILE"
require_file "$RESULTS_ROOT/weissbart_three_way_test_metrics.csv" "combined long metrics"
cp "$RESULTS_ROOT/weissbart_three_way_test_metrics.csv" "$RESULTS_ROOT/ds004408_three_way_test_metrics.csv"
require_file "$RESULTS_ROOT/megxl_paper_metrics_summary.csv" "aggregate metrics"
require_file "$RESULTS_ROOT/megxl_pairwise_welch_tests.csv" "Welch tests"
require_file "$RESULTS_ROOT/megxl_paper_report_manifest.json" "combined report"
echo "$RUN_ID" > "$RESULTS_ROOT/COMPLETED"; rm -f "$PID_FILE"

echo "ds004408 three-way fine-tuning completed."
echo "Word alignment: $WORD_ALIGNED_OUTPUT/summary.json"
echo "Metrics: $RESULTS_ROOT/ds004408_three_way_test_metrics.csv"
echo "Results: $RESULTS_ROOT"
