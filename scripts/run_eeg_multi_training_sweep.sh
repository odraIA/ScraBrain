#!/usr/bin/env bash
set -uo pipefail

# Sequential EEG multi-dataset training sweep.
# Runs beta/gamma/high-gamma frequency bands across selected tokenizers using Docker Compose.
# It writes a global final_results.txt with links to every run folder.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.eeg-train.yml)
SERVICE="${EEG_TRAIN_SERVICE:-eeg_train_multi}"
CONFIG_NAME="${EEG_CONFIG_NAME:-train_criss_cross_eeg_multi_feq}"
PYTHON_MODULE="${EEG_TRAIN_MODULE:-brainstorm.train_criss_cross_eeg_multi}"
GPU="${EEG_GPU:-0,1}"
SEED="${EEG_SEED:-42}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ -n "${EEG_GPU_COUNT:-}" ]]; then
  GPU_COUNT="$EEG_GPU_COUNT"
elif [[ "$GPU" == "all" ]]; then
  GPU_COUNT="all"
elif [[ "$GPU" == *","* ]]; then
  GPU_COUNT="$(awk -F',' '{print NF}' <<< "$GPU")"
elif [[ -z "${EEG_GPU:-}" ]]; then
  GPU_COUNT="2"
else
  GPU_COUNT="1"
fi

if [[ -n "${EEG_TRAINER_DEVICES:-}" ]]; then
  TRAINER_DEVICES="$EEG_TRAINER_DEVICES"
elif [[ "$GPU_COUNT" == "all" ]]; then
  TRAINER_DEVICES="auto"
else
  TRAINER_DEVICES="$GPU_COUNT"
fi

if [[ -n "${EEG_TRAINER_STRATEGY:-}" ]]; then
  TRAINER_STRATEGY="$EEG_TRAINER_STRATEGY"
elif [[ "$TRAINER_DEVICES" =~ ^[0-9]+$ && "$TRAINER_DEVICES" -gt 1 ]]; then
  TRAINER_STRATEGY="fsdp"
else
  TRAINER_STRATEGY="null"
fi

export EEG_GPU="$GPU"
export EEG_GPU_COUNT="$GPU_COUNT"

# By default run every experiment twice: scratch and MEG-XL initialization.
# To run only one mode:
#   EEG_INIT_MODES="scratch" bash scripts/run_eeg_multi_training_sweep.sh
#   EEG_INIT_MODES="pretrained" bash scripts/run_eeg_multi_training_sweep.sh
read -r -a INIT_MODES <<< "${EEG_INIT_MODES:-scratch pretrained}"

# Path to the MEG-XL checkpoint used when init_mode=pretrained.
CRISS_CROSS_CHECKPOINT="${CRISS_CROSS_CHECKPOINT:-./checkpoints/baseline/meg-xl-med.ckpt}"

MAX_STEPS="${EEG_MAX_STEPS:-}"          # optional quick-test cap, e.g. EEG_MAX_STEPS=100
NUM_EPOCHS="${EEG_NUM_EPOCHS:-}"        # optional override, e.g. EEG_NUM_EPOCHS=5
LIMIT="${EEG_SWEEP_LIMIT:-0}"           # 0 = no limit
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

if [[ -n "${EEG_VAL_CHECK_INTERVAL:-}" ]]; then
  VAL_CHECK_INTERVAL="$EEG_VAL_CHECK_INTERVAL"
elif [[ "$MAX_STEPS" =~ ^[0-9]+$ && "$MAX_STEPS" -gt 0 && "$MAX_STEPS" -lt 500 ]]; then
  VAL_CHECK_INTERVAL="$MAX_STEPS"
else
  VAL_CHECK_INTERVAL="500"
fi

if [[ -n "${EEG_CHECKPOINT_EVERY_N_TRAIN_STEPS:-}" ]]; then
  CHECKPOINT_EVERY_N_TRAIN_STEPS="$EEG_CHECKPOINT_EVERY_N_TRAIN_STEPS"
elif [[ "$MAX_STEPS" =~ ^[0-9]+$ && "$MAX_STEPS" -gt 0 && "$MAX_STEPS" -lt 5000 ]]; then
  CHECKPOINT_EVERY_N_TRAIN_STEPS="$MAX_STEPS"
else
  CHECKPOINT_EVERY_N_TRAIN_STEPS="5000"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_ROOT="results/eeg_multi_training_sweep/${STAMP}"
mkdir -p "$SWEEP_ROOT"

# Default bands: beta/gamma/high-gamma. Override with EEG_BANDS="beta_13_24 gamma_30_55".
read -r -a BANDS <<< "${EEG_BANDS:-beta_13_24 beta_gamma_13_45 low_gamma_30_45 gamma_30_55 high_gamma_70_120}"

# Default tokenizers. BrainOmni/BrainTokenizer runs require their functional
# tokenizer model code to be wired into brainstorm/neuro_tokenizers/factory.py.
read -r -a TOKENIZERS <<< "${EEG_TOKENIZERS:-biocodec}"

band_values() {
  case "$1" in
    beta_13_24)        echo "50.0 13.0 24.0" ;;
    beta_gamma_13_45)  echo "100.0 13.0 45.0" ;;
    low_gamma_30_45)   echo "100.0 30.0 45.0" ;;
    gamma_30_55)       echo "128.0 30.0 55.0" ;;
    high_gamma_70_120) echo "250.0 70.0 120.0" ;;
    *)
      echo "Unknown band '$1'. Valid: beta_13_24 beta_gamma_13_45 low_gamma_30_45 gamma_30_55 high_gamma_70_120" >&2
      return 1
      ;;
  esac
}

tokenizer_overrides() {
  case "$1" in
    biocodec)
      echo "model.tokenizer_name=biocodec model.tokenizer_variant=default model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/biocodec_ckpt.pt model.vocab_size=256 model.num_quantizers=6 model.num_quantizers_used=6 model.tokenizer_downsample_ratio=12 model.overlap_ratio=0.0"
      ;;
    brainomni_base)
      echo "model.tokenizer_name=brainomni_base model.tokenizer_variant=base model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/base/BrainOmni.pt model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/base/BrainOmni.pt model.tokenizer_config_path=./brainstorm/neuro_tokenizers/base/model_cfg.json model.vocab_size=512 model.num_quantizers=4 model.num_quantizers_used=4 model.tokenizer_downsample_ratio=64 model.overlap_ratio=0.25"
      ;;
    brainomni_tiny)
      echo "model.tokenizer_name=brainomni_tiny model.tokenizer_variant=tiny model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/tiny/BrainOmni.pt model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/tiny/BrainOmni.pt model.tokenizer_config_path=./brainstorm/neuro_tokenizers/tiny/model_cfg.json model.vocab_size=512 model.num_quantizers=4 model.num_quantizers_used=4 model.tokenizer_downsample_ratio=64 model.overlap_ratio=0.25"
      ;;
    braintokenizer)
      echo "model.tokenizer_name=braintokenizer model.tokenizer_variant=default model.tokenizer_checkpoint=./brainstorm/neuro_tokenizers/braintokenizer/BrainTokenizer.pt model.tokenizer_ckpt=./brainstorm/neuro_tokenizers/braintokenizer/BrainTokenizer.pt model.tokenizer_config_path=./brainstorm/neuro_tokenizers/braintokenizer/model_cfg.json model.vocab_size=512 model.num_quantizers=4 model.num_quantizers_used=4 model.tokenizer_downsample_ratio=64 model.overlap_ratio=0.0"
      ;;
    *)
      echo "Unknown tokenizer '$1'. Valid: biocodec brainomni_base brainomni_tiny braintokenizer" >&2
      return 1
      ;;
  esac
}

init_overrides() {
  local init_mode="$1"

  case "$init_mode" in
    scratch)
      echo "model.train_from_scratch=true model.use_promoted_checkpoint=false"
      ;;
    pretrained)
      echo "model.train_from_scratch=false model.use_promoted_checkpoint=false model.criss_cross_checkpoint=${CRISS_CROSS_CHECKPOINT}"
      ;;
    *)
      echo "Unknown init mode '$init_mode'. Valid: scratch pretrained" >&2
      return 1
      ;;
  esac
}

append_result() {
  local status="$1"
  local exp="$2"
  local exit_code="$3"
  local run_dir="$4"
  local checkpoint_dir="$5"
  local final_txt="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$status" "$exit_code" "$exp" "$run_dir" "$checkpoint_dir" "$final_txt" >> "$SWEEP_ROOT/runs.tsv"
}

{
  echo -e "status\texit_code\texperiment\trun_dir\tcheckpoint_dir\tfinal_results_txt"
} > "$SWEEP_ROOT/runs.tsv"

cat > "$SWEEP_ROOT/sweep_metadata.txt" <<EOF
EEG multi-dataset training sweep
Started: ${STAMP}
Root: ${SWEEP_ROOT}
Bands: ${BANDS[*]}
Tokenizers: ${TOKENIZERS[*]}
Init modes: ${INIT_MODES[*]}
MEG checkpoint: ${CRISS_CROSS_CHECKPOINT}
GPU: ${GPU}
GPU count: ${GPU_COUNT}
Trainer devices: ${TRAINER_DEVICES}
Trainer strategy: ${TRAINER_STRATEGY}
Validation interval: ${VAL_CHECK_INTERVAL}
Checkpoint interval: ${CHECKPOINT_EVERY_N_TRAIN_STEPS}
WANDB_MODE: ${WANDB_MODE}
Config: ${CONFIG_NAME}
Module: ${PYTHON_MODULE}
EOF

echo "Building Docker image/service '${SERVICE}'..."
docker compose "${COMPOSE_FILES[@]}" build "$SERVICE"
build_status=$?
if [[ $build_status -ne 0 ]]; then
  echo "Docker build failed with exit code ${build_status}" | tee "$SWEEP_ROOT/final_results.txt"
  exit $build_status
fi

run_idx=0
success_count=0
fail_count=0

for init_mode in "${INIT_MODES[@]}"; do
  init="$(init_overrides "$init_mode")" || exit 1

  for band in "${BANDS[@]}"; do
    read -r target_sfreq l_freq h_freq <<< "$(band_values "$band")" || exit 1

    for tokenizer in "${TOKENIZERS[@]}"; do
      tok_overrides="$(tokenizer_overrides "$tokenizer")" || exit 1

      run_idx=$((run_idx + 1))
      if [[ "$LIMIT" != "0" && "$run_idx" -gt "$LIMIT" ]]; then
        echo "Reached EEG_SWEEP_LIMIT=${LIMIT}; stopping."
        break 3
      fi

      exp="eeg_multi_${band}_${tokenizer}_${init_mode}_seed${SEED}"
      run_dir="logs/eeg_multi_training/${exp}"
      checkpoint_dir="checkpoints/eeg_multi_training/${exp}"
      host_run_dir="$SWEEP_ROOT/${exp}"
      mkdir -p "$host_run_dir"

      cmd=(
        uv run --no-sync python -m "$PYTHON_MODULE"
        --config-name "$CONFIG_NAME"
        "data.target_sfreq=${target_sfreq}"
        "model.sampling_rate=${target_sfreq%.*}"
        "data.l_freq=${l_freq}"
        "data.h_freq=${h_freq}"
        "data.cache_dir=./data/cache/eeg_multi_training/${band}_${tokenizer}"
        "logging.experiment_name=${exp}"
        "logging.save_dir=./logs/eeg_multi_training"
        "checkpoint.save_dir=./checkpoints/eeg_multi_training"
        "trainer.devices=${TRAINER_DEVICES}"
        "trainer.strategy=${TRAINER_STRATEGY}"
        "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
        "checkpoint.every_n_train_steps=${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
        "seed=${SEED}"
        $tok_overrides
        $init
      )

      if [[ -n "$MAX_STEPS" ]]; then
        cmd+=("training.max_steps=${MAX_STEPS}" "training.num_epochs=null")
      elif [[ -n "$NUM_EPOCHS" ]]; then
        cmd+=("training.num_epochs=${NUM_EPOCHS}" "training.max_steps=null")
      fi

      printf '%q ' "${cmd[@]}" > "$host_run_dir/command.txt"
      echo >> "$host_run_dir/command.txt"

      echo
      echo "================================================================================"
      echo "[$run_idx] Running ${exp}"
      echo "================================================================================"
      echo "Run artifacts will be in: ${run_dir}"
      echo "Checkpoints will be in:   ${checkpoint_dir}"

      container_name="scrabrain_${exp//[^a-zA-Z0-9_.-]/_}"
      docker compose "${COMPOSE_FILES[@]}" run \
        --rm \
        --name "$container_name" \
        -e EEG_GPU="$GPU" \
        -e WANDB_MODE="$WANDB_MODE" \
        "$SERVICE" \
        bash -lc "${cmd[*]}" \
        2>&1 | tee "$host_run_dir/stdout_stderr.log"

      exit_code=${PIPESTATUS[0]}
      final_txt="${run_dir}/final_results.txt"

      if [[ $exit_code -eq 0 ]]; then
        success_count=$((success_count + 1))
        append_result "OK" "$exp" "$exit_code" "$run_dir" "$checkpoint_dir" "$final_txt"
      else
        fail_count=$((fail_count + 1))
        append_result "FAILED" "$exp" "$exit_code" "$run_dir" "$checkpoint_dir" "$final_txt"
        if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
          echo "Stopping because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}."
          break 3
        fi
      fi
    done
  done
done

{
  echo "EEG Multi-Dataset Training Sweep Final Results"
  echo "================================================"
  echo "Started: ${STAMP}"
  echo "Finished: $(date +%Y%m%d_%H%M%S)"
  echo "Sweep root: ${SWEEP_ROOT}"
  echo "Total launched: $((success_count + fail_count))"
  echo "Succeeded: ${success_count}"
  echo "Failed: ${fail_count}"
  echo
  echo "Run table: ${SWEEP_ROOT}/runs.tsv"
  echo "Per-run host logs: ${SWEEP_ROOT}/<experiment>/stdout_stderr.log"
  echo "Per-run command files: ${SWEEP_ROOT}/<experiment>/command.txt"
  echo
  echo "Inside repo, each successful/failed training run also writes:"
  echo "  logs/eeg_multi_training/<experiment>/config_resolved.yaml"
  echo "  logs/eeg_multi_training/<experiment>/stdout_stderr.log"
  echo "  logs/eeg_multi_training/<experiment>/epoch_metrics.csv"
  echo "  logs/eeg_multi_training/<experiment>/epoch_metrics.jsonl"
  echo "  logs/eeg_multi_training/<experiment>/final_results.txt"
  echo "  logs/eeg_multi_training/<experiment>/final_results.json"
  echo "  checkpoints/eeg_multi_training/<experiment>/checkpoint_best.pt"
  echo "  checkpoints/eeg_multi_training/<experiment>/checkpoint_latest.pt"
  echo
  echo "Summary table:"
  column -t -s $'\t' "$SWEEP_ROOT/runs.tsv" 2>/dev/null || cat "$SWEEP_ROOT/runs.tsv"
} | tee "$SWEEP_ROOT/final_results.txt"

if [[ $fail_count -gt 0 ]]; then
  exit 2
fi
exit 0
