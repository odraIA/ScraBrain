#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential three-way Alice EEG word-aligned fine-tuning comparison on one GPU.
#
# Runs, in order:
#   1. Randomly initialized CrissCross architecture.
#   2. EEG checkpoint originally trained from scratch (reading -> listening).
#   3. EEG checkpoint initialized from pretrained MEG-XL (reading -> listening).
#
# Every run uses the same no-leak text split, validation checkpoint selection,
# and final test protocol as the Weissbart experiment. In addition to MEG-XL's
# balanced Top-10 metrics over 50/250 words, it evaluates Top-1 and Top-10 over
# the 601-word Alice vocabulary for comparison with Chen et al.
#
# Run from the repository root:
#   bash scripts/run_alice_three_way_finetuning.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_alice_three_way_finetuning.sh"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ALICE_ROOT="${ALICE_ROOT:-./datasets/alice_eeg}"
BIOCODEC_CHECKPOINT="${BIOCODEC_CHECKPOINT:-./brainstorm/neuro_tokenizers/biocodec_ckpt.pt}"
MEGXL_ARCH_CHECKPOINT="${MEGXL_ARCH_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"

SCRATCH_EEG_CHECKPOINT="${SCRATCH_EEG_CHECKPOINT:-./checkpoints/eeg_full_band_reading_then_listening_compare/20260624_182700/eeg_full_band_0p1_50_fixed50_50hz_biocodec_from_scratch_listening_seed42/checkpoint_best.pt}"
PRETRAINED_EEG_CHECKPOINT="${PRETRAINED_EEG_CHECKPOINT:-./checkpoints/eeg_full_band_reading_then_listening_compare/20260624_182700/eeg_full_band_0p1_50_fixed50_50hz_biocodec_pretrained_listening_seed42/checkpoint_best.pt}"

TRAIN_PCT="${TRAIN_PCT:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-./results/alice_three_way/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-./logs/word_classification_alice_eeg/$RUN_ID}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/word_classification_alice_eeg/$RUN_ID}"
HYDRA_ROOT="${HYDRA_ROOT:-./logs/hydra/alice_three_way/$RUN_ID}"
MASTER_LOG="${MASTER_LOG:-$ROOT_DIR/logs/alice_three_way_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/alice_three_way_${RUN_ID}.pid}"
CURRENT_CONTAINER_FILE="$RESULTS_ROOT/current_container.txt"
STATUS_FILE="$RESULTS_ROOT/runs.tsv"

mkdir -p \
  "$RESULTS_ROOT" \
  "$LOG_ROOT" \
  "$CHECKPOINT_ROOT" \
  "$HYDRA_ROOT" \
  "$(dirname "$MASTER_LOG")"

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: Missing $description: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: Missing $description: $path" >&2
    exit 2
  fi
}

preflight() {
  command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is not available" >&2
    exit 2
  }
  require_dir "$ALICE_ROOT" "Alice EEG dataset directory"
  require_file "$ALICE_ROOT/AliceChapterOne-EEG.csv" "Alice word annotations"
  compgen -G "$ALICE_ROOT/S*.vhdr" >/dev/null || {
    echo "ERROR: No Alice BrainVision S*.vhdr files found in $ALICE_ROOT" >&2
    exit 2
  }
  require_file "$BIOCODEC_CHECKPOINT" "BioCodec checkpoint"
  require_file "$MEGXL_ARCH_CHECKPOINT" "MEG-XL architecture checkpoint"
  require_file "$SCRATCH_EEG_CHECKPOINT" "EEG checkpoint originally trained from scratch"
  require_file "$PRETRAINED_EEG_CHECKPOINT" "EEG checkpoint initialized from pretrained MEG-XL"
}

preflight

# Launcher mode. The worker itself is detached, and each experiment is a
# detached Docker container that is explicitly awaited before the next starts.
if [[ "${ALICE_THREE_WAY_WORKER:-0}" != "1" ]]; then
  nohup env \
    ALICE_THREE_WAY_WORKER=1 \
    RUN_ID="$RUN_ID" \
    MASTER_LOG="$MASTER_LOG" \
    PID_FILE="$PID_FILE" \
    GPU="$GPU" \
    WANDB_MODE="$WANDB_MODE" \
    BUILD_IMAGE="$BUILD_IMAGE" \
    ALICE_ROOT="$ALICE_ROOT" \
    BIOCODEC_CHECKPOINT="$BIOCODEC_CHECKPOINT" \
    MEGXL_ARCH_CHECKPOINT="$MEGXL_ARCH_CHECKPOINT" \
    SCRATCH_EEG_CHECKPOINT="$SCRATCH_EEG_CHECKPOINT" \
    PRETRAINED_EEG_CHECKPOINT="$PRETRAINED_EEG_CHECKPOINT" \
    TRAIN_PCT="$TRAIN_PCT" \
    NUM_EPOCHS="$NUM_EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    RESULTS_ROOT="$RESULTS_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
    HYDRA_ROOT="$HYDRA_ROOT" \
    bash "$SCRIPT_PATH" \
    >> "$MASTER_LOG" 2>&1 < /dev/null &

  ORCHESTRATOR_PID=$!
  printf '%s\n' "$ORCHESTRATOR_PID" > "$PID_FILE"
  printf '%s\n' "$RUN_ID" > "$ROOT_DIR/alice_three_way.latest"

  echo "Alice three-way pipeline launched in background."
  echo "Run ID: $RUN_ID"
  echo "Orchestrator PID: $ORCHESTRATOR_PID"
  echo "PID file: $PID_FILE"
  echo "Log: $MASTER_LOG"
  echo "Results: $RESULTS_ROOT"
  exit 0
fi

printf '%s\n' "$$" > "$PID_FILE"
printf 'order\tlabel\tcontainer\tstatus\texit_code\tstarted_at\tfinished_at\n' > "$STATUS_FILE"

if [[ "$BUILD_IMAGE" == "1" || "$BUILD_IMAGE" == "true" ]]; then
  echo "Building Docker image..."
  docker compose build eval_eeg_listening
fi

CURRENT_EXPERIMENT=""
CURRENT_CONTAINER=""

on_error() {
  local status=$?
  echo "ERROR: experiment ${CURRENT_EXPERIMENT:-unknown} failed with exit code $status" >&2
  if [[ -n "$CURRENT_CONTAINER" ]]; then
    echo "Container retained for inspection: $CURRENT_CONTAINER" >&2
    echo "Inspect with: docker logs --tail 200 $CURRENT_CONTAINER" >&2
  fi
  exit "$status"
}
trap on_error ERR

append_status() {
  local order="$1"
  local label="$2"
  local container="$3"
  local status="$4"
  local exit_code="$5"
  local started_at="$6"
  local finished_at="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$order" "$label" "$container" "$status" "$exit_code" "$started_at" "$finished_at" \
    >> "$STATUS_FILE"
}

run_experiment() {
  local order="$1"
  local label="$2"
  local train_from_scratch="$3"
  local initialization_checkpoint="$4"
  local build_combined_report="$5"

  local experiment_name="alice_${label}_${RUN_ID}"
  local container_name="alice_${label}_${RUN_ID}"
  local save_dir="$LOG_ROOT/$label"
  local checkpoint_dir="$CHECKPOINT_ROOT/$label"
  local hydra_dir="$HYDRA_ROOT/$label"
  local extra_container_env=()
  local started_at
  local finished_at
  local container_exit_code

  CURRENT_EXPERIMENT="$experiment_name"
  CURRENT_CONTAINER="$container_name"
  mkdir -p "$save_dir" "$checkpoint_dir" "$hydra_dir"

  if docker container inspect "$container_name" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container_name")" == "true" ]]; then
      echo "ERROR: container already running: $container_name" >&2
      exit 3
    fi
    docker rm "$container_name" >/dev/null
  fi

  if [[ "$build_combined_report" == "1" ]]; then
    extra_container_env+=(
      -e "ALICE_COMPARISON_RUNS=random_init=$LOG_ROOT/random_init;eeg_from_scratch=$LOG_ROOT/eeg_from_scratch;eeg_pretrained=$LOG_ROOT/eeg_pretrained"
      -e "ALICE_COMPARISON_OUTPUT=$RESULTS_ROOT"
    )
  fi

  started_at="$(date --iso-8601=seconds)"
  printf '%s\n' "$container_name" > "$CURRENT_CONTAINER_FILE"

  echo
  echo "================================================================================"
  echo "START: $experiment_name"
  echo "Container: $container_name"
  echo "GPU: $GPU"
  echo "Initialization checkpoint: $initialization_checkpoint"
  echo "Random initialization: $train_from_scratch"
  echo "Fine-tuning data: Alice chapter-one word-aligned listening EEG (N=33 main set)"
  echo "MEG-XL metrics: balanced Top-10 over 50 and 250 words"
  echo "Paper comparison: Top-1 and Top-10 over 601 words"
  echo "Final evaluation: best validation checkpoint on the held-out test split"
  echo "================================================================================"

  env \
    EEG_GPU="$GPU" \
    WANDB_MODE="$WANDB_MODE" \
    docker compose run -d --no-deps \
      --name "$container_name" \
      -e "NVIDIA_VISIBLE_DEVICES=$GPU" \
      -e "WANDB_MODE=$WANDB_MODE" \
      "${extra_container_env[@]}" \
      eval_eeg_listening \
      uv run --no-sync python -m brainstorm.evaluate_criss_cross_word_classification_alice_reported \
        --config-name=eval_criss_cross_word_classification_alice_eeg \
        "model.train_from_scratch=$train_from_scratch" \
        model.use_promoted_checkpoint=false \
        model.promoted_checkpoint=null \
        "model.criss_cross_checkpoint=$initialization_checkpoint" \
        model.tokenizer_name=biocodec \
        "model.tokenizer_checkpoint=$BIOCODEC_CHECKPOINT" \
        "data.root=$ALICE_ROOT" \
        data.subject_selection=main \
        "data.train_pct=$TRAIN_PCT" \
        "training.num_epochs=$NUM_EPOCHS" \
        "training.batch_size=$BATCH_SIZE" \
        "training.num_workers=$NUM_WORKERS" \
        'evaluation.retrieval_set_sizes=[50,250,601]' \
        evaluation.k=10 \
        "logging.experiment_name=$experiment_name" \
        "logging.save_dir=$save_dir" \
        "logging.checkpoint_dir=$checkpoint_dir" \
        "hydra.run.dir=$hydra_dir"

  echo "Detached container launched: $container_name"
  docker logs --follow "$container_name" || true

  container_exit_code="$(docker wait "$container_name")"
  finished_at="$(date --iso-8601=seconds)"

  if [[ "$container_exit_code" != "0" ]]; then
    append_status \
      "$order" "$label" "$container_name" "FAILED" "$container_exit_code" \
      "$started_at" "$finished_at"
    echo "ERROR: $container_name exited with code $container_exit_code" >&2
    echo "The failed container has been retained for inspection." >&2
    exit "$container_exit_code"
  fi

  require_file "$save_dir/final_results.json" "final test results for $experiment_name"
  require_file "$save_dir/paper_test_metrics.csv" "MEG-XL metrics for $experiment_name"
  require_file "$save_dir/paper_report_manifest.json" "MEG-XL report manifest for $experiment_name"
  require_file "$save_dir/alice_reference_comparison.csv" "Alice paper comparison for $experiment_name"
  require_file "$save_dir/alice_reference_comparison.json" "Alice paper comparison metadata for $experiment_name"
  require_file "$save_dir/final_test_top10_accuracy.png" "final test graph for $experiment_name"
  require_file "$checkpoint_dir/checkpoint_best.pt" "best fine-tuned checkpoint for $experiment_name"

  append_status \
    "$order" "$label" "$container_name" "COMPLETED" "$container_exit_code" \
    "$started_at" "$finished_at"

  docker rm "$container_name" >/dev/null
  CURRENT_CONTAINER=""

  echo "COMPLETED: $experiment_name"
  echo "Results: $save_dir/final_results.json"
  echo "MEG-XL metrics: $save_dir/paper_test_metrics.csv"
  echo "Alice paper comparison: $save_dir/alice_reference_comparison.csv"
  echo "Best checkpoint: $checkpoint_dir/checkpoint_best.pt"
}

run_experiment 1 "random_init" true "$MEGXL_ARCH_CHECKPOINT" 0
run_experiment 2 "eeg_from_scratch" false "$SCRATCH_EEG_CHECKPOINT" 0
run_experiment 3 "eeg_pretrained" false "$PRETRAINED_EEG_CHECKPOINT" 1

CURRENT_EXPERIMENT="summary"
rm -f "$CURRENT_CONTAINER_FILE"

require_file "$RESULTS_ROOT/alice_three_way_test_metrics.csv" "three-way long-format metrics"
require_file "$RESULTS_ROOT/alice_reference_three_way_comparison.csv" "three-way Alice paper comparison"
require_file "$RESULTS_ROOT/megxl_paper_metrics_summary.csv" "MEG-XL aggregate metrics"
require_file "$RESULTS_ROOT/megxl_pairwise_welch_tests.csv" "Welch comparison table"
require_file "$RESULTS_ROOT/megxl_figure3_top10_retrieval50.png" "MEG-XL Figure 3-style graph"
require_file "$RESULTS_ROOT/megxl_figure6_top10_retrieval250.png" "MEG-XL Figure 6-style graph"
require_file "$RESULTS_ROOT/alice_megxl_report_manifest.json" "combined Alice report manifest"

echo
echo "================================================================================"
echo "ALL THREE ALICE RUNS COMPLETED"
echo "Run ID: $RUN_ID"
echo "Status: $STATUS_FILE"
echo "Metrics: $RESULTS_ROOT/alice_three_way_test_metrics.csv"
echo "Paper comparison: $RESULTS_ROOT/alice_reference_three_way_comparison.csv"
echo "Aggregate mean/SEM: $RESULTS_ROOT/megxl_paper_metrics_summary.csv"
echo "Welch tests: $RESULTS_ROOT/megxl_pairwise_welch_tests.csv"
echo "Top-50 graph: $RESULTS_ROOT/megxl_figure3_top10_retrieval50.png"
echo "Top-250 graph: $RESULTS_ROOT/megxl_figure6_top10_retrieval250.png"
echo "================================================================================"
