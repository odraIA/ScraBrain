#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare ds004408 once and compare four initializations in parallel:
#   1. random CrissCross initialization;
#   2. EEG curriculum trained from scratch, using EEG embedding row 2;
#   3. MEG-XL -> EEG curriculum, using dedicated EEG embedding row 2;
#   4. MEG-XL -> EEG curriculum, reusing MEG embedding row 1 for EEG.
#
# ds004408 always remains physical EEG sensor type 2. The per-run embedding id
# only selects the sensor_type_layer row used by that checkpoint.
#
# By default, GPUs that look idle at launch are detected automatically and two
# fine-tunings are assigned to each GPU. With two idle GPUs, all four runs start
# concurrently. Use GPU_LIST=0,1 to select devices explicitly, or
# JOBS_PER_GPU=1 to use only one process per GPU.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/run_ds004408_four_way_finetuning.sh"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
FREE_GPU_MAX_MEMORY_MIB="${FREE_GPU_MAX_MEMORY_MIB:-1024}"
FREE_GPU_MAX_UTILIZATION="${FREE_GPU_MAX_UTILIZATION:-10}"
OMP_NUM_THREADS_PER_JOB="${OMP_NUM_THREADS_PER_JOB:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

DS004408_ROOT="${DS004408_ROOT:-./datasets/OpenNeuroEEG_ds004408}"
BIOCODEC_CHECKPOINT="${BIOCODEC_CHECKPOINT:-./brainstorm/neuro_tokenizers/biocodec_ckpt.pt}"
MEGXL_ARCH_CHECKPOINT="${MEGXL_ARCH_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
CURRICULUM_ROOT="${CURRICULUM_ROOT:-./checkpoints/eeg_language_curriculum_three_models/20260629_004853}"
FROM_SCRATCH_EEG_CHECKPOINT="${FROM_SCRATCH_EEG_CHECKPOINT:-$CURRICULUM_ROOT/eeg_curriculum_from_scratch_language_seed42/checkpoint_best.pt}"
MEGXL_EEG2_CHECKPOINT="${MEGXL_EEG2_CHECKPOINT:-$CURRICULUM_ROOT/eeg_curriculum_megxl_eeg2_language_seed42/checkpoint_best.pt}"
MEGXL_EEG1_CHECKPOINT="${MEGXL_EEG1_CHECKPOINT:-$CURRICULUM_ROOT/eeg_curriculum_megxl_eeg1_language_seed42/checkpoint_best.pt}"

TRAIN_PCT="${TRAIN_PCT:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREPARE_WORD_ALIGNED="${PREPARE_WORD_ALIGNED:-1}"
WARM_WORD_ALIGNED_CACHE="${WARM_WORD_ALIGNED_CACHE:-1}"

RESULTS_ROOT="${RESULTS_ROOT:-./results/ds004408_four_way/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-./logs/word_classification_ds004408_four_way/$RUN_ID}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/word_classification_ds004408_four_way/$RUN_ID}"
HYDRA_ROOT="${HYDRA_ROOT:-./logs/hydra/ds004408_four_way/$RUN_ID}"
WORD_ALIGNED_OUTPUT="${WORD_ALIGNED_OUTPUT:-$RESULTS_ROOT/word_aligned}"
WORD_ALIGNED_CACHE="${WORD_ALIGNED_CACHE:-./data/cache/ds004408_word_aligned_v2}"
MASTER_LOG="${MASTER_LOG:-$ROOT_DIR/logs/ds004408_four_way_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-$ROOT_DIR/ds004408_four_way_${RUN_ID}.pid}"
CURRENT_CONTAINERS_FILE="$RESULTS_ROOT/current_containers.tsv"
STATUS_FILE="$RESULTS_ROOT/runs.tsv"

JOB_LABELS=(random_init eeg_from_scratch megxl_eeg2 megxl_eeg1)
JOB_TRAIN_FROM_SCRATCH=(true false false false)
JOB_CHECKPOINTS=(
  "$MEGXL_ARCH_CHECKPOINT"
  "$FROM_SCRATCH_EEG_CHECKPOINT"
  "$MEGXL_EEG2_CHECKPOINT"
  "$MEGXL_EEG1_CHECKPOINT"
)
JOB_EMBEDDING_IDS=(2 2 2 1)

declare -a AVAILABLE_GPUS=()
declare -a GPU_SLOTS=()
declare -A CONTAINER_BY_LABEL=()
declare -A GPU_BY_LABEL=()
declare -A EMBEDDING_BY_LABEL=()
declare -A ORDER_BY_LABEL=()
declare -A STARTED_BY_LABEL=()
declare -A SAVE_DIR_BY_LABEL=()
declare -A CHECKPOINT_DIR_BY_LABEL=()
declare -A LOG_PID_BY_LABEL=()

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "$CHECKPOINT_ROOT" "$HYDRA_ROOT" \
  "$WORD_ALIGNED_OUTPUT" "$(dirname "$MASTER_LOG")"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: Missing $2: $1" >&2; exit 2; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR: Missing $2: $1" >&2; exit 2; }
}

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

resolve_available_gpus() {
  local gpu index memory_used utilization round
  if [[ "$GPU_LIST" == "auto" ]]; then
    while IFS=',' read -r index memory_used utilization; do
      index="$(trim "$index")"
      memory_used="$(trim "$memory_used")"
      utilization="$(trim "$utilization")"
      if (( memory_used <= FREE_GPU_MAX_MEMORY_MIB && utilization <= FREE_GPU_MAX_UTILIZATION )); then
        AVAILABLE_GPUS+=("$index")
      else
        echo "Skipping busy GPU $index: memory=${memory_used} MiB, utilization=${utilization}%"
      fi
    done < <(
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
  else
    IFS=',' read -r -a requested_gpus <<< "$GPU_LIST"
    for gpu in "${requested_gpus[@]}"; do
      gpu="$(trim "$gpu")"
      [[ -n "$gpu" ]] || continue
      nvidia-smi -i "$gpu" >/dev/null 2>&1 || {
        echo "ERROR: Invalid or unavailable GPU index: $gpu" >&2
        exit 2
      }
      AVAILABLE_GPUS+=("$gpu")
    done
  fi

  ((${#AVAILABLE_GPUS[@]} > 0)) || {
    echo "ERROR: No free GPUs found." >&2
    echo "Set GPU_LIST=0,1 explicitly or adjust FREE_GPU_MAX_MEMORY_MIB/FREE_GPU_MAX_UTILIZATION." >&2
    exit 2
  }

  for ((round = 0; round < JOBS_PER_GPU; round++)); do
    for gpu in "${AVAILABLE_GPUS[@]}"; do
      GPU_SLOTS+=("$gpu")
    done
  done

  ((${#GPU_SLOTS[@]} > 0)) || {
    echo "ERROR: JOBS_PER_GPU must be at least 1" >&2
    exit 2
  }

  echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
  echo "Concurrent slots: ${#GPU_SLOTS[@]} (${JOBS_PER_GPU} job(s) per GPU)"
}

preflight() {
  command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is not available" >&2
    exit 2
  }
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi is not available" >&2
    exit 2
  }
  [[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: JOBS_PER_GPU must be a positive integer" >&2
    exit 2
  }
  [[ "$FREE_GPU_MAX_MEMORY_MIB" =~ ^[0-9]+$ ]] || {
    echo "ERROR: FREE_GPU_MAX_MEMORY_MIB must be a non-negative integer" >&2
    exit 2
  }
  [[ "$FREE_GPU_MAX_UTILIZATION" =~ ^[0-9]+$ ]] || {
    echo "ERROR: FREE_GPU_MAX_UTILIZATION must be a non-negative integer" >&2
    exit 2
  }

  require_dir "$DS004408_ROOT" "ds004408 dataset directory"
  compgen -G "$DS004408_ROOT/sub-*/eeg/*_eeg.vhdr" >/dev/null || {
    echo "ERROR: No ds004408 BrainVision files under $DS004408_ROOT" >&2
    exit 2
  }
  compgen -G "$DS004408_ROOT/stimuli/*.TextGrid" >/dev/null || {
    echo "ERROR: No materialized TextGrids. Run scripts/clone_openneuro_ds004408.sh" >&2
    exit 2
  }
  require_file "$BIOCODEC_CHECKPOINT" "BioCodec checkpoint"
  require_file "$MEGXL_ARCH_CHECKPOINT" "MEG-XL architecture checkpoint"
  require_file "$FROM_SCRATCH_EEG_CHECKPOINT" "curriculum checkpoint from scratch"
  require_file "$MEGXL_EEG2_CHECKPOINT" "curriculum checkpoint MEG-XL/eeg2"
  require_file "$MEGXL_EEG1_CHECKPOINT" "curriculum checkpoint MEG-XL/eeg1"

  resolve_available_gpus
}
preflight

if [[ "${DS004408_FOUR_WAY_WORKER:-0}" != "1" ]]; then
  nohup env DS004408_FOUR_WAY_WORKER=1 RUN_ID="$RUN_ID" MASTER_LOG="$MASTER_LOG" \
    PID_FILE="$PID_FILE" GPU_LIST="$(IFS=,; echo "${AVAILABLE_GPUS[*]}")" \
    JOBS_PER_GPU="$JOBS_PER_GPU" OMP_NUM_THREADS_PER_JOB="$OMP_NUM_THREADS_PER_JOB" \
    FREE_GPU_MAX_MEMORY_MIB="$FREE_GPU_MAX_MEMORY_MIB" \
    FREE_GPU_MAX_UTILIZATION="$FREE_GPU_MAX_UTILIZATION" \
    WANDB_MODE="$WANDB_MODE" BUILD_IMAGE="$BUILD_IMAGE" \
    DS004408_ROOT="$DS004408_ROOT" BIOCODEC_CHECKPOINT="$BIOCODEC_CHECKPOINT" \
    MEGXL_ARCH_CHECKPOINT="$MEGXL_ARCH_CHECKPOINT" CURRICULUM_ROOT="$CURRICULUM_ROOT" \
    FROM_SCRATCH_EEG_CHECKPOINT="$FROM_SCRATCH_EEG_CHECKPOINT" \
    MEGXL_EEG2_CHECKPOINT="$MEGXL_EEG2_CHECKPOINT" \
    MEGXL_EEG1_CHECKPOINT="$MEGXL_EEG1_CHECKPOINT" TRAIN_PCT="$TRAIN_PCT" \
    NUM_EPOCHS="$NUM_EPOCHS" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    PREPARE_WORD_ALIGNED="$PREPARE_WORD_ALIGNED" \
    WARM_WORD_ALIGNED_CACHE="$WARM_WORD_ALIGNED_CACHE" RESULTS_ROOT="$RESULTS_ROOT" \
    LOG_ROOT="$LOG_ROOT" CHECKPOINT_ROOT="$CHECKPOINT_ROOT" HYDRA_ROOT="$HYDRA_ROOT" \
    WORD_ALIGNED_OUTPUT="$WORD_ALIGNED_OUTPUT" WORD_ALIGNED_CACHE="$WORD_ALIGNED_CACHE" \
    bash "$SCRIPT_PATH" >> "$MASTER_LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  echo "$RUN_ID" > "$ROOT_DIR/ds004408_four_way.latest"
  echo "ds004408 four-way pipeline launched. PID: $(cat "$PID_FILE")"
  echo "GPUs: ${AVAILABLE_GPUS[*]} | jobs per GPU: $JOBS_PER_GPU"
  echo "Log: $MASTER_LOG"
  echo "Results: $RESULTS_ROOT"
  exit 0
fi

echo $$ > "$PID_FILE"
printf 'order\tlabel\tembedding_id\tgpu\tcontainer\tstatus\texit_code\tstarted_at\tfinished_at\n' > "$STATUS_FILE"
printf 'label\tgpu\tcontainer\n' > "$CURRENT_CONTAINERS_FILE"

if [[ "$BUILD_IMAGE" == "1" || "$BUILD_IMAGE" == "true" ]]; then
  docker compose build eval_eeg_listening
fi

on_error() {
  local status=$?
  echo "ERROR: ds004408 four-way pipeline failed ($status)" >&2
  if [[ -s "$CURRENT_CONTAINERS_FILE" ]]; then
    echo "Containers launched in this run:" >&2
    tail -n +2 "$CURRENT_CONTAINERS_FILE" >&2 || true
  fi
  exit "$status"
}
trap on_error ERR

append_status() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$STATUS_FILE"
}

prepare_word_aligned() {
  [[ "$PREPARE_WORD_ALIGNED" == "1" || "$PREPARE_WORD_ALIGNED" == "true" ]] || return 0
  local started finished warm_flag="--no-warm-cache"
  local prep_gpu="${AVAILABLE_GPUS[0]}"
  started="$(date --iso-8601=seconds)"
  [[ "$WARM_WORD_ALIGNED_CACHE" == "1" || "$WARM_WORD_ALIGNED_CACHE" == "true" ]] && warm_flag="--warm-cache"

  echo "Preparing ds004408 word alignment once on GPU $prep_gpu"
  env EEG_GPU="$prep_gpu" WANDB_MODE="$WANDB_MODE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS_PER_JOB" \
    docker compose run --rm --no-deps \
      -e "NVIDIA_VISIBLE_DEVICES=$prep_gpu" \
      -e "WANDB_MODE=$WANDB_MODE" \
      -e "OMP_NUM_THREADS=$OMP_NUM_THREADS_PER_JOB" \
      eval_eeg_listening \
      uv run --no-sync python scripts/prepare_ds004408_word_aligned.py \
        --root "$DS004408_ROOT" --cache-dir "$WORD_ALIGNED_CACHE" \
        --output-dir "$WORD_ALIGNED_OUTPUT" --eeg-sensor-type eeg \
        --montage-name biosemi128 --drop-bad-channels "$warm_flag"

  require_file "$WORD_ALIGNED_OUTPUT/summary.json" "word-aligned summary"
  require_file "$WORD_ALIGNED_OUTPUT/word_aligned_manifest.csv" "word-aligned manifest"
  require_file "$WORD_ALIGNED_OUTPUT/alignment_report.json" "alignment report"
  finished="$(date --iso-8601=seconds)"
  append_status 0 prepare_word_aligned 2 "$prep_gpu" docker-compose-run-rm COMPLETED 0 "$started" "$finished"
}

launch_experiment() {
  local order="$1" label="$2" train_from_scratch="$3" init_checkpoint="$4"
  local embedding_id="$5" gpu="$6"
  local experiment="ds004408_${label}_${RUN_ID}" container="ds004408_${label}_${RUN_ID}"
  local save_dir="$LOG_ROOT/$label" checkpoint_dir="$CHECKPOINT_ROOT/$label"
  local hydra_dir="$HYDRA_ROOT/$label" started

  if [[ "$embedding_id" != "1" && "$embedding_id" != "2" ]]; then
    echo "ERROR: Unsupported EEG embedding id for $label: $embedding_id" >&2
    exit 2
  fi

  mkdir -p "$save_dir" "$checkpoint_dir" "$hydra_dir"
  if docker container inspect "$container" >/dev/null 2>&1; then
    [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]] || {
      echo "ERROR: container already running: $container" >&2
      exit 3
    }
    docker rm "$container" >/dev/null
  fi

  started="$(date --iso-8601=seconds)"
  echo "START $experiment | GPU=$gpu | physical_sensor=eeg:2 | embedding=$embedding_id | checkpoint=$init_checkpoint"

  env EEG_GPU="$gpu" WANDB_MODE="$WANDB_MODE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS_PER_JOB" \
    docker compose run -d --no-deps --name "$container" \
      -e "NVIDIA_VISIBLE_DEVICES=$gpu" \
      -e "WANDB_MODE=$WANDB_MODE" \
      -e "OMP_NUM_THREADS=$OMP_NUM_THREADS_PER_JOB" \
      -e "EEG_SENSOR_EMBEDDING_TYPE_ID=$embedding_id" \
      eval_eeg_listening \
      uv run --no-sync python -m scripts.evaluate_ds004408_word_classification \
        --config-name=ds004408_word_finetuning \
        "model.train_from_scratch=$train_from_scratch" model.use_promoted_checkpoint=false \
        model.promoted_checkpoint=null "model.criss_cross_checkpoint=$init_checkpoint" \
        "model.eeg_sensor_embedding_type_id=$embedding_id" \
        model.tokenizer_name=biocodec "model.tokenizer_checkpoint=$BIOCODEC_CHECKPOINT" \
        "data.root=$DS004408_ROOT" "data.cache_dir=$WORD_ALIGNED_CACHE" \
        data.eeg_sensor_type=eeg data.montage_name=biosemi128 data.drop_bad_channels=true \
        "data.train_pct=$TRAIN_PCT" "training.num_epochs=$NUM_EPOCHS" \
        "training.batch_size=$BATCH_SIZE" "training.num_workers=$NUM_WORKERS" \
        'evaluation.retrieval_set_sizes=[50,250]' evaluation.k=10 \
        "logging.experiment_name=$experiment" "logging.save_dir=$save_dir" \
        "logging.checkpoint_dir=$checkpoint_dir" "hydra.run.dir=$hydra_dir"

  CONTAINER_BY_LABEL["$label"]="$container"
  GPU_BY_LABEL["$label"]="$gpu"
  EMBEDDING_BY_LABEL["$label"]="$embedding_id"
  ORDER_BY_LABEL["$label"]="$order"
  STARTED_BY_LABEL["$label"]="$started"
  SAVE_DIR_BY_LABEL["$label"]="$save_dir"
  CHECKPOINT_DIR_BY_LABEL["$label"]="$checkpoint_dir"
  printf '%s\t%s\t%s\n' "$label" "$gpu" "$container" >> "$CURRENT_CONTAINERS_FILE"

  docker logs --follow "$container" 2>&1 | sed -u "s/^/[$label|gpu$gpu] /" &
  LOG_PID_BY_LABEL["$label"]=$!
}

validate_experiment_outputs() {
  local label="$1"
  local save_dir="${SAVE_DIR_BY_LABEL[$label]}"
  local checkpoint_dir="${CHECKPOINT_DIR_BY_LABEL[$label]}"
  local missing=0 path description

  while IFS='|' read -r path description; do
    if [[ ! -f "$path" ]]; then
      echo "ERROR: Missing $description for $label: $path" >&2
      missing=1
    fi
  done <<EOF
$save_dir/final_results.json|final results
$save_dir/paper_test_metrics.csv|paper metrics
$save_dir/paper_report_manifest.json|report manifest
$checkpoint_dir/checkpoint_best.pt|best checkpoint
EOF

  ((missing == 0))
}

wait_experiment() {
  local label="$1"
  local container="${CONTAINER_BY_LABEL[$label]}"
  local gpu="${GPU_BY_LABEL[$label]}"
  local embedding_id="${EMBEDDING_BY_LABEL[$label]}"
  local order="${ORDER_BY_LABEL[$label]}"
  local started="${STARTED_BY_LABEL[$label]}"
  local exit_code finished status

  exit_code="$(docker wait "$container")"
  wait "${LOG_PID_BY_LABEL[$label]}" || true
  finished="$(date --iso-8601=seconds)"

  if [[ "$exit_code" == "0" ]] && validate_experiment_outputs "$label"; then
    status="COMPLETED"
    docker rm "$container" >/dev/null
  else
    status="FAILED"
    echo "ERROR: $container exited with $exit_code; keeping it for inspection" >&2
  fi

  append_status "$order" "$label" "$embedding_id" "$gpu" "$container" \
    "$status" "$exit_code" "$started" "$finished"

  [[ "$status" == "COMPLETED" ]]
}

run_all_experiments() {
  local total_jobs="${#JOB_LABELS[@]}"
  local slot_count="${#GPU_SLOTS[@]}"
  local batch_start job_index slot_index label gpu
  local -a batch_labels=()
  local failures=0

  echo "Launching $total_jobs experiments with up to $slot_count concurrent containers"

  for ((batch_start = 0; batch_start < total_jobs; batch_start += slot_count)); do
    batch_labels=()
    for ((slot_index = 0; slot_index < slot_count; slot_index++)); do
      job_index=$((batch_start + slot_index))
      ((job_index < total_jobs)) || break

      label="${JOB_LABELS[$job_index]}"
      gpu="${GPU_SLOTS[$slot_index]}"
      launch_experiment \
        "$((job_index + 1))" \
        "$label" \
        "${JOB_TRAIN_FROM_SCRATCH[$job_index]}" \
        "${JOB_CHECKPOINTS[$job_index]}" \
        "${JOB_EMBEDDING_IDS[$job_index]}" \
        "$gpu"
      batch_labels+=("$label")
    done

    echo "Batch running: ${batch_labels[*]}"
    for label in "${batch_labels[@]}"; do
      if ! wait_experiment "$label"; then
        failures=$((failures + 1))
      fi
    done
  done

  ((failures == 0)) || {
    echo "ERROR: $failures fine-tuning job(s) failed" >&2
    return 1
  }
}

generate_comparison_report() {
  local report_gpu="${AVAILABLE_GPUS[0]}"
  echo "Generating combined four-way report"

  env EEG_GPU="$report_gpu" WANDB_MODE="$WANDB_MODE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS_PER_JOB" \
    docker compose run --rm --no-deps \
      -e "NVIDIA_VISIBLE_DEVICES=$report_gpu" \
      -e "WANDB_MODE=$WANDB_MODE" \
      -e "OMP_NUM_THREADS=$OMP_NUM_THREADS_PER_JOB" \
      eval_eeg_listening \
      uv run --no-sync python -m brainstorm.megxl_test_reporting compare \
        --run "random_init=$LOG_ROOT/random_init" \
        --run "eeg_from_scratch=$LOG_ROOT/eeg_from_scratch" \
        --run "megxl_eeg2=$LOG_ROOT/megxl_eeg2" \
        --run "megxl_eeg1=$LOG_ROOT/megxl_eeg1" \
        --output-dir "$RESULTS_ROOT" \
        --retrieval-sizes 50 250 \
        --top-k 10

  require_file "$RESULTS_ROOT/weissbart_three_way_test_metrics.csv" "combined long metrics"
  cp "$RESULTS_ROOT/weissbart_three_way_test_metrics.csv" \
    "$RESULTS_ROOT/ds004408_four_way_test_metrics.csv"
  require_file "$RESULTS_ROOT/megxl_paper_metrics_summary.csv" "aggregate metrics"
  require_file "$RESULTS_ROOT/megxl_pairwise_welch_tests.csv" "Welch tests"
  require_file "$RESULTS_ROOT/megxl_paper_report_manifest.json" "combined report"
}

prepare_word_aligned
run_all_experiments
generate_comparison_report

rm -f "$CURRENT_CONTAINERS_FILE"
echo "$RUN_ID" > "$RESULTS_ROOT/COMPLETED"
rm -f "$PID_FILE"

echo "ds004408 four-way fine-tuning completed."
echo "GPUs used: ${AVAILABLE_GPUS[*]}"
echo "Word alignment: $WORD_ALIGNED_OUTPUT/summary.json"
echo "Metrics: $RESULTS_ROOT/ds004408_four_way_test_metrics.csv"
echo "Results: $RESULTS_ROOT"
