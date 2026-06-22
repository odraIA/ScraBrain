#!/usr/bin/env bash
set -uo pipefail

# Two-GPU queue for staged continuous EEG pre-training:
#   1) reading EEG
#   2) listening EEG initialized from the best reading checkpoint
#
# The default matrix contains:
#   - 48 fixed-50-Hz controls: 6 bands x 4 tokenizers x 2 initializations
#   - 32 Nyquist-aware repetitions: 4 affected bands x 4 tokenizers x 2 initializations
#   - 80 experiment pipelines total, each with two training stages (160 stage runs)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

COMPOSE_FILE="${EEG_COMPOSE_FILE:-${ROOT_DIR}/docker-compose.eeg-reading-listening.yml}"
SERVICE="${EEG_TRAIN_SERVICE:-eeg_train_reading_listening}"
PYTHON_MODULE="${EEG_TRAIN_MODULE:-brainstorm.train_criss_cross_eeg_continuous}"
READING_CONFIG="${EEG_READING_CONFIG:-train_criss_cross_eeg_reading_continuous}"
LISTENING_CONFIG="${EEG_LISTENING_CONFIG:-train_criss_cross_eeg_listening_continuous}"
SEED="${EEG_SEED:-42}"
WANDB_MODE="${WANDB_MODE:-offline}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
AUTO_BATCH="${EEG_AUTO_BATCH:-true}"
DEFAULT_BATCH_SIZE="${EEG_DEFAULT_BATCH_SIZE:-4}"
BATCH_CANDIDATES_RAW="${EEG_BATCH_CANDIDATES:-16 12 8 6 4 2 1}"
GRADIENT_CHECKPOINTING="${EEG_GRADIENT_CHECKPOINTING:-false}"
NUM_WORKERS="${EEG_NUM_WORKERS:-6}"
VAL_CHECK_INTERVAL="${EEG_VAL_CHECK_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${EEG_CHECKPOINT_EVERY_N_TRAIN_STEPS:-5000}"
CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"
MAX_STEPS="${EEG_MAX_STEPS:-}"
NUM_EPOCHS="${EEG_NUM_EPOCHS:-}"
LIMIT="${EEG_SWEEP_LIMIT:-0}"
DRY_RUN="${EEG_DRY_RUN:-0}"

read -r -a GPUS <<< "${EEG_GPUS:-0 1}"
read -r -a INIT_MODES <<< "${EEG_INIT_MODES:-scratch pretrained}"
read -r -a TOKENIZERS <<< "${EEG_TOKENIZERS:-biocodec brainomni_base brainomni_tiny braintokenizer}"
read -r -a BATCH_CANDIDATES <<< "$BATCH_CANDIDATES_RAW"

if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "EEG_GPUS must contain at least one GPU id." >&2
  exit 2
fi

if [[ -n "$MAX_STEPS" && -n "$NUM_EPOCHS" ]]; then
  echo "Set only one of EEG_MAX_STEPS or EEG_NUM_EPOCHS." >&2
  exit 2
fi

STAMP="${EEG_SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${EEG_SWEEP_ROOT:-results/eeg_reading_listening_sweep/${STAMP}}"
QUEUE_FILE="${SWEEP_ROOT}/jobs.tsv"
NEXT_FILE="${SWEEP_ROOT}/next_job.txt"
QUEUE_LOCK="${SWEEP_ROOT}/queue.lock"
RESULTS_LOCK="${SWEEP_ROOT}/results.lock"
BATCH_LOCK="${SWEEP_ROOT}/batch_sizes.lock"
BATCH_CACHE="${SWEEP_ROOT}/batch_sizes.tsv"
RUNS_FILE="${SWEEP_ROOT}/runs.tsv"

mkdir -p \
  "$SWEEP_ROOT" \
  data/cache/eeg_reading_listening_continuous \
  logs/eeg_reading_listening_training \
  checkpoints/eeg_reading_listening_training \
  results/eeg_reading_listening_sweep \
  wandb

sanitize() {
  printf '%s' "$1" | tr -c '[:alnum:]_.-' '_'
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_oom_failure() {
  local exit_code="$1"
  local log_path="$2"
  if [[ "$exit_code" == "137" || "$exit_code" == "134" ]]; then
    return 0
  fi
  grep -Eiq \
    'CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: an illegal memory access|DefaultCPUAllocator:.*allocate memory' \
    "$log_path" 2>/dev/null
}

gpu_memory_mib() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | head -n1 | tr -d '[:space:]'
}

tokenizer_overrides() {
  case "$1" in
    biocodec)
      printf '%s\n' \
        'model.tokenizer_name=biocodec' \
        'model.tokenizer_variant=default' \
        'model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt' \
        'model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt' \
        'model.tokenizer_config_path=null' \
        'model.vocab_size=256' \
        'model.num_quantizers=6' \
        'model.num_quantizers_used=6' \
        'model.tokenizer_downsample_ratio=12' \
        'model.overlap_ratio=0.0'
      ;;
    brainomni_base)
      printf '%s\n' \
        'model.tokenizer_name=brainomni_base' \
        'model.tokenizer_variant=base' \
        'model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/base/BrainOmni.pt' \
        'model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/base/BrainOmni.pt' \
        'model.tokenizer_config_path=./brainstorm/neuro_tokenizers/base/model_cfg.json' \
        'model.vocab_size=512' \
        'model.num_quantizers=4' \
        'model.num_quantizers_used=4' \
        'model.tokenizer_downsample_ratio=64' \
        'model.overlap_ratio=0.25'
      ;;
    brainomni_tiny)
      printf '%s\n' \
        'model.tokenizer_name=brainomni_tiny' \
        'model.tokenizer_variant=tiny' \
        'model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/tiny/BrainOmni.pt' \
        'model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/tiny/BrainOmni.pt' \
        'model.tokenizer_config_path=./brainstorm/neuro_tokenizers/tiny/model_cfg.json' \
        'model.vocab_size=512' \
        'model.num_quantizers=4' \
        'model.num_quantizers_used=4' \
        'model.tokenizer_downsample_ratio=64' \
        'model.overlap_ratio=0.25'
      ;;
    braintokenizer)
      printf '%s\n' \
        'model.tokenizer_name=braintokenizer' \
        'model.tokenizer_variant=default' \
        'model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/braintokenizer/BrainTokenizer.pt' \
        'model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/braintokenizer/BrainTokenizer.pt' \
        'model.tokenizer_config_path=./brainstorm/neuro_tokenizers/braintokenizer/model_cfg.json' \
        'model.vocab_size=512' \
        'model.num_quantizers=4' \
        'model.num_quantizers_used=4' \
        'model.tokenizer_downsample_ratio=64' \
        'model.overlap_ratio=0.0'
      ;;
    *)
      echo "Unknown tokenizer '$1'." >&2
      return 1
      ;;
  esac
}

write_job_matrix() {
  local count=0
  printf 'job_id\tband\tprofile\ttarget_sfreq\tl_freq\th_freq\ttokenizer\tinitialization\n' > "$QUEUE_FILE"

  emit_job() {
    local band="$1" profile="$2" sfreq="$3" low="$4" high="$5" tokenizer="$6" init="$7"
    count=$((count + 1))
    if [[ "$LIMIT" != "0" && "$count" -gt "$LIMIT" ]]; then
      return 0
    fi
    local job_id
    job_id="$(printf '%03d' "$count")_${band}_${profile}_${tokenizer}_${init}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$job_id" "$band" "$profile" "$sfreq" "$low" "$high" "$tokenizer" "$init" >> "$QUEUE_FILE"
  }

  local init tokenizer
  for init in "${INIT_MODES[@]}"; do
    for tokenizer in "${TOKENIZERS[@]}"; do
      # Fixed-50-Hz controls. For bands above 25 Hz this deliberately tests the
      # current MEG-XL filter-then-resample behavior without preserving the full band.
      emit_job alpha_8_12 fixed50 50 8 12 "$tokenizer" "$init"
      emit_job beta_13_24 fixed50 50 13 24 "$tokenizer" "$init"
      emit_job beta_gamma_13_45 fixed50 50 13 45 "$tokenizer" "$init"
      emit_job low_gamma_30_45 fixed50 50 30 45 "$tokenizer" "$init"
      emit_job gamma_30_55 fixed50 50 30 55 "$tokenizer" "$init"
      emit_job high_gamma_70_120 fixed50 50 70 120 "$tokenizer" "$init"

      # Nyquist-aware repetitions only for the four bands affected at 50 Hz.
      emit_job beta_gamma_13_45 nyquist 100 13 45 "$tokenizer" "$init"
      emit_job low_gamma_30_45 nyquist 100 30 45 "$tokenizer" "$init"
      emit_job gamma_30_55 nyquist 128 30 55 "$tokenizer" "$init"
      emit_job high_gamma_70_120 nyquist 250 70 120 "$tokenizer" "$init"
    done
  done

  echo 0 > "$NEXT_FILE"
  printf 'batch_key\tbatch_size\tgpu_memory_mib\tprobe_gpu\ttimestamp\n' > "$BATCH_CACHE"
  printf 'job_id\tgpu\tstatus\treading_batch\tlistening_batch\treading_experiment\tlistening_experiment\tmessage\n' > "$RUNS_FILE"
}

claim_job() {
  local line index
  exec 9>"$QUEUE_LOCK"
  flock 9
  index="$(cat "$NEXT_FILE")"
  line="$(sed -n "$((index + 2))p" "$QUEUE_FILE")"
  if [[ -z "$line" ]]; then
    flock -u 9
    exec 9>&-
    return 1
  fi
  echo $((index + 1)) > "$NEXT_FILE"
  flock -u 9
  exec 9>&-
  printf '%s\n' "$line"
}

append_run_result() {
  exec 8>"$RESULTS_LOCK"
  flock 8
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$RUNS_FILE"
  flock -u 8
  exec 8>&-
}

lookup_batch_size() {
  local key="$1"
  awk -F '\t' -v key="$key" 'NR > 1 && $1 == key {value=$2} END {if (value != "") print value}' "$BATCH_CACHE"
}

record_batch_size() {
  local key="$1" batch="$2" memory="$3" gpu="$4"
  exec 7>"$BATCH_LOCK"
  flock 7
  if ! awk -F '\t' -v key="$key" 'NR > 1 && $1 == key {found=1} END {exit !found}' "$BATCH_CACHE"; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$key" "$batch" "$memory" "$gpu" "$(date -Iseconds)" >> "$BATCH_CACHE"
  fi
  flock -u 7
  exec 7>&-
}

base_command() {
  local config="$1" experiment="$2" sfreq="$3" low="$4" high="$5" tokenizer="$6" batch="$7"
  local -n output_ref="$8"
  local tok
  mapfile -t tok < <(tokenizer_overrides "$tokenizer") || return 1

  output_ref=(
    uv run --no-sync python -m "$PYTHON_MODULE"
    --config-name "$config"
    "data.target_sfreq=${sfreq}"
    "model.sampling_rate=${sfreq%.*}"
    "data.l_freq=${low}"
    "data.h_freq=${high}"
    'data.cache_dir=./data/cache/eeg_reading_listening_continuous'
    "training.batch_size=${batch}"
    'trainer.devices=1'
    'trainer.strategy=auto'
    "model.use_gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
    "logging.experiment_name=${experiment}"
    'logging.save_dir=./logs/eeg_reading_listening_training'
    'checkpoint.save_dir=./checkpoints/eeg_reading_listening_training'
    "seed=${SEED}"
  )
  output_ref+=("${tok[@]}")
}

run_docker_command() {
  local gpu="$1" container_name="$2" log_path="$3"
  shift 3
  local -a command=("$@")
  local quoted
  printf -v quoted '%q ' "${command[@]}"

  EEG_GPU="$gpu" EEG_GPU_COUNT=1 WANDB_MODE="$WANDB_MODE" \
    docker compose -f "$COMPOSE_FILE" run \
      --rm --no-deps \
      --name "$container_name" \
      -e "WANDB_MODE=${WANDB_MODE}" \
      "$SERVICE" \
      bash -lc "$quoted" \
      2>&1 | tee "$log_path" >&2
  return "${PIPESTATUS[0]}"
}

probe_batch_size() {
  local stage="$1" config="$2" gpu="$3" profile="$4" band="$5" sfreq="$6" low="$7" high="$8" tokenizer="$9"
  local memory key cached candidate probe_exp probe_dir probe_log container_name status
  local -a command

  memory="$(gpu_memory_mib "$gpu")"
  memory="${memory:-unknown}"
  key="${stage}_${band}_${profile}_${sfreq}hz_${tokenizer}_${memory}mib"
  cached="$(lookup_batch_size "$key")"
  if [[ -n "$cached" ]]; then
    printf '%s\n' "$cached"
    return 0
  fi

  if ! is_true "$AUTO_BATCH"; then
    record_batch_size "$key" "$DEFAULT_BATCH_SIZE" "$memory" "$gpu"
    printf '%s\n' "$DEFAULT_BATCH_SIZE"
    return 0
  fi

  for candidate in "${BATCH_CANDIDATES[@]}"; do
    probe_exp="__batch_probe__${stage}_${band}_${profile}_${sfreq}hz_${tokenizer}_b${candidate}_gpu${gpu}"
    probe_dir="${SWEEP_ROOT}/batch_probes/${probe_exp}"
    probe_log="${probe_dir}/stdout_stderr.log"
    container_name="$(sanitize "scrabrain_${probe_exp}")"
    mkdir -p "$probe_dir"

    base_command "$config" "$probe_exp" "$sfreq" "$low" "$high" "$tokenizer" "$candidate" command || return 1
    command+=(
      'training.max_steps=2'
      'training.num_epochs=null'
      'training.num_workers=0'
      'training.persistent_workers=false'
      'trainer.val_check_interval=2'
      'checkpoint.save_top_k=0'
      'checkpoint.save_last=false'
      'logging.wandb_project='
      'model.train_from_scratch=true'
      'model.use_promoted_checkpoint=false'
    )

    printf '%q ' "${command[@]}" > "${probe_dir}/command.txt"
    echo >> "${probe_dir}/command.txt"
    echo "[GPU ${gpu}] Probing ${stage} batch=${candidate} for ${band}/${profile}/${tokenizer}" >&2

    run_docker_command "$gpu" "$container_name" "$probe_log" "${command[@]}"
    status=$?
    if [[ $status -eq 0 ]]; then
      record_batch_size "$key" "$candidate" "$memory" "$gpu"
      printf '%s\n' "$candidate"
      return 0
    fi
    if ! is_oom_failure "$status" "$probe_log"; then
      echo "Batch probe failed for a non-OOM reason; see ${probe_log}" >&2
      return "$status"
    fi
    echo "[GPU ${gpu}] batch=${candidate} does not fit; trying a smaller batch." >&2
  done

  echo "No batch candidate fits for ${stage}/${band}/${profile}/${tokenizer} on GPU ${gpu}." >&2
  return 1
}

next_smaller_batch() {
  local current="$1" candidate seen=0
  for candidate in "${BATCH_CANDIDATES[@]}"; do
    if [[ "$seen" == "1" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    if [[ "$candidate" == "$current" ]]; then
      seen=1
    fi
  done
  if [[ "$current" -gt 1 ]]; then
    printf '%s\n' $((current - 1))
    return 0
  fi
  return 1
}

run_stage() {
  local stage="$1" config="$2" gpu="$3" experiment="$4" sfreq="$5" low="$6" high="$7" tokenizer="$8" init_mode="$9" promoted_checkpoint="${10:-}" initial_batch="${11}"
  local batch="$initial_batch" stage_dir log_path container_name status smaller
  local -a command

  while true; do
    stage_dir="${SWEEP_ROOT}/stages/${experiment}/batch_${batch}"
    log_path="${stage_dir}/stdout_stderr.log"
    container_name="$(sanitize "scrabrain_${experiment}_b${batch}")"
    mkdir -p "$stage_dir"

    base_command "$config" "$experiment" "$sfreq" "$low" "$high" "$tokenizer" "$batch" command || return 1
    command+=(
      "training.num_workers=${NUM_WORKERS}"
      'training.persistent_workers=true'
      "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
      "checkpoint.every_n_train_steps=${CHECKPOINT_INTERVAL}"
    )
    if [[ -n "$MAX_STEPS" ]]; then
      command+=("training.max_steps=${MAX_STEPS}" 'training.num_epochs=null')
    elif [[ -n "$NUM_EPOCHS" ]]; then
      command+=("training.num_epochs=${NUM_EPOCHS}" 'training.max_steps=null')
    fi

    if [[ "$stage" == "reading" ]]; then
      case "$init_mode" in
        scratch)
          command+=(
            'model.train_from_scratch=true'
            'model.use_promoted_checkpoint=false'
          )
          ;;
        pretrained)
          command+=(
            'model.train_from_scratch=false'
            'model.use_promoted_checkpoint=false'
            "model.criss_cross_checkpoint=${CRISS_CROSS_CHECKPOINT}"
          )
          ;;
        *)
          echo "Unknown initialization '$init_mode'." >&2
          return 2
          ;;
      esac
    else
      command+=(
        'model.train_from_scratch=false'
        'model.use_promoted_checkpoint=true'
        "model.promoted_checkpoint=${promoted_checkpoint}"
      )
    fi

    printf '%q ' "${command[@]}" > "${stage_dir}/command.txt"
    echo >> "${stage_dir}/command.txt"
    echo "[GPU ${gpu}] Starting ${stage}: ${experiment} (batch=${batch})" >&2

    run_docker_command "$gpu" "$container_name" "$log_path" "${command[@]}"
    status=$?
    if [[ $status -eq 0 ]]; then
      printf '%s\n' "$batch"
      return 0
    fi

    if is_oom_failure "$status" "$log_path" && smaller="$(next_smaller_batch "$batch")"; then
      echo "[GPU ${gpu}] ${experiment} OOM at batch=${batch}; retrying with batch=${smaller}." >&2
      rm -rf "logs/eeg_reading_listening_training/${experiment}" \
             "checkpoints/eeg_reading_listening_training/${experiment}"
      batch="$smaller"
      continue
    fi

    echo "[GPU ${gpu}] ${experiment} failed; see ${log_path}" >&2
    return "$status"
  done
}

find_stage_checkpoint() {
  local experiment="$1"
  local checkpoint_root="checkpoints/eeg_reading_listening_training/${experiment}"
  if [[ -s "${checkpoint_root}/checkpoint_best.pt" ]]; then
    printf './%s\n' "${checkpoint_root}/checkpoint_best.pt"
  elif [[ -s "${checkpoint_root}/checkpoint_latest.pt" ]]; then
    printf './%s\n' "${checkpoint_root}/checkpoint_latest.pt"
  else
    return 1
  fi
}

stage_already_completed() {
  local experiment="$1"
  local result="logs/eeg_reading_listening_training/${experiment}/final_results.json"
  [[ -s "$result" ]] || return 1
  python3 - "$result" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "completed" else 1)
PY
}

process_job() {
  local gpu="$1" job_line="$2"
  local job_id band profile sfreq low high tokenizer init_mode
  local reading_exp listening_exp reading_batch listening_batch reading_checkpoint
  local status message

  IFS=$'\t' read -r job_id band profile sfreq low high tokenizer init_mode <<< "$job_line"
  reading_exp="eeg_${band}_${profile}_${sfreq}hz_${tokenizer}_${init_mode}_reading_seed${SEED}"
  listening_exp="eeg_${band}_${profile}_${sfreq}hz_${tokenizer}_${init_mode}_listening_seed${SEED}"

  echo
  echo "================================================================================"
  echo "[GPU ${gpu}] ${job_id}: ${band} ${profile}, ${tokenizer}, ${init_mode}"
  echo "================================================================================"

  reading_batch="$(probe_batch_size reading "$READING_CONFIG" "$gpu" "$profile" "$band" "$sfreq" "$low" "$high" "$tokenizer")" || {
    append_run_result "$job_id" "$gpu" FAILED '' '' "$reading_exp" "$listening_exp" 'reading batch probe failed'
    return 1
  }

  if stage_already_completed "$reading_exp" && reading_checkpoint="$(find_stage_checkpoint "$reading_exp")"; then
    echo "[GPU ${gpu}] Reusing completed reading stage ${reading_exp}."
  else
    rm -rf "logs/eeg_reading_listening_training/${reading_exp}" \
           "checkpoints/eeg_reading_listening_training/${reading_exp}"
    reading_batch="$(run_stage reading "$READING_CONFIG" "$gpu" "$reading_exp" "$sfreq" "$low" "$high" "$tokenizer" "$init_mode" '' "$reading_batch")" || {
      append_run_result "$job_id" "$gpu" FAILED "$reading_batch" '' "$reading_exp" "$listening_exp" 'reading stage failed'
      return 1
    }
    reading_checkpoint="$(find_stage_checkpoint "$reading_exp")" || {
      append_run_result "$job_id" "$gpu" FAILED "$reading_batch" '' "$reading_exp" "$listening_exp" 'reading checkpoint missing'
      return 1
    }
  fi

  listening_batch="$(probe_batch_size listening "$LISTENING_CONFIG" "$gpu" "$profile" "$band" "$sfreq" "$low" "$high" "$tokenizer")" || {
    append_run_result "$job_id" "$gpu" FAILED "$reading_batch" '' "$reading_exp" "$listening_exp" 'listening batch probe failed'
    return 1
  }

  if stage_already_completed "$listening_exp"; then
    echo "[GPU ${gpu}] Reusing completed listening stage ${listening_exp}."
  else
    rm -rf "logs/eeg_reading_listening_training/${listening_exp}" \
           "checkpoints/eeg_reading_listening_training/${listening_exp}"
    listening_batch="$(run_stage listening "$LISTENING_CONFIG" "$gpu" "$listening_exp" "$sfreq" "$low" "$high" "$tokenizer" "$init_mode" "$reading_checkpoint" "$listening_batch")" || {
      append_run_result "$job_id" "$gpu" FAILED "$reading_batch" "$listening_batch" "$reading_exp" "$listening_exp" 'listening stage failed'
      return 1
    }
  fi

  status=OK
  message="reading_checkpoint=${reading_checkpoint}"
  append_run_result "$job_id" "$gpu" "$status" "$reading_batch" "$listening_batch" "$reading_exp" "$listening_exp" "$message"
  return 0
}

worker() {
  local gpu="$1" job_line
  while job_line="$(claim_job)"; do
    if ! process_job "$gpu" "$job_line"; then
      if ! is_true "$CONTINUE_ON_ERROR"; then
        echo "[GPU ${gpu}] Stopping worker because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}." >&2
        return 1
      fi
    fi
  done
  echo "[GPU ${gpu}] Queue empty."
}

write_job_matrix
TOTAL_JOBS=$(( $(wc -l < "$QUEUE_FILE") - 1 ))

cat > "${SWEEP_ROOT}/sweep_metadata.txt" <<EOF_META
Staged continuous EEG sweep
Started: $(date -Iseconds)
Sweep root: ${SWEEP_ROOT}
Jobs: ${TOTAL_JOBS}
Training stages: $((TOTAL_JOBS * 2))
GPUs: ${GPUS[*]}
Bands: alpha_8_12 beta_13_24 beta_gamma_13_45 low_gamma_30_45 gamma_30_55 high_gamma_70_120
Profiles: fixed50 for all bands; Nyquist-aware repeats for beta-gamma/low-gamma/gamma/high-gamma
Tokenizers: ${TOKENIZERS[*]}
Initializations: ${INIT_MODES[*]}
Default batch: ${DEFAULT_BATCH_SIZE}
Auto batch: ${AUTO_BATCH}
Batch candidates: ${BATCH_CANDIDATES[*]}
Gradient checkpointing: ${GRADIENT_CHECKPOINTING}
Data cache (unchanged): ./data/cache/eeg_reading_listening_continuous
Reading config: ${READING_CONFIG}
Listening config: ${LISTENING_CONFIG}
MEG-XL checkpoint: ${CRISS_CROSS_CHECKPOINT}
EOF_META

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run complete: ${TOTAL_JOBS} jobs (${TOTAL_JOBS} reading + ${TOTAL_JOBS} listening stages)."
  echo "Plan: ${QUEUE_FILE}"
  exit 0
fi

echo "Building Docker service ${SERVICE}..."
docker compose -f "$COMPOSE_FILE" build "$SERVICE" || exit $?

echo "Launching ${#GPUS[@]} queue workers for ${TOTAL_JOBS} staged experiments."
worker_pids=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  worker_pids+=("$!")
done

cleanup() {
  local pid
  for pid in "${worker_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

worker_failure=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || worker_failure=1
done

ok_count="$(awk -F '\t' 'NR > 1 && $3 == "OK" {count++} END {print count+0}' "$RUNS_FILE")"
failed_count="$(awk -F '\t' 'NR > 1 && $3 == "FAILED" {count++} END {print count+0}' "$RUNS_FILE")"

cat > "${SWEEP_ROOT}/final_results.txt" <<EOF_FINAL
Staged continuous EEG sweep results
===================================
Finished: $(date -Iseconds)
Sweep root: ${SWEEP_ROOT}
Jobs planned: ${TOTAL_JOBS}
Jobs completed: ${ok_count}
Jobs failed: ${failed_count}
Run table: ${RUNS_FILE}
Batch-size table: ${BATCH_CACHE}
Per-stage launcher logs: ${SWEEP_ROOT}/stages/<experiment>/batch_<n>/stdout_stderr.log
Training outputs: ./logs/eeg_reading_listening_training/<experiment>
Checkpoints: ./checkpoints/eeg_reading_listening_training/<experiment>
EOF_FINAL

cat "${SWEEP_ROOT}/final_results.txt"
exit "$worker_failure"
