ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRAIN_YML="${EEG_COMPOSE_FILE:-$ROOT/docker-compose.eeg-reading-listening.yml}"
PRE_YML="${EEG_PREPROCESS_COMPOSE_FILE:-$ROOT/docker-compose.eeg-preprocess.yml}"
STAMP="${EEG_SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED="${EEG_SEED:-42}"
CACHE="${EEG_CACHE_DIR:-./data/cache/eeg_preprocessed}"
RUN_ROOT="${EEG_SWEEP_ROOT:-results/eeg_full_band_reading_then_listening_compare/$STAMP}"
LOG_ROOT="${EEG_TRAIN_LOG_ROOT:-./logs/eeg_full_band_reading_then_listening_compare/$STAMP}"
CKPT_ROOT="${EEG_CHECKPOINT_ROOT:-./checkpoints/eeg_full_band_reading_then_listening_compare/$STAMP}"
STAGING="${EEG_STAGING_CACHE_ROOT:-./data/cache/eeg_full_band_reading_then_listening_staging/$STAMP}"
READ_BATCH="${EEG_READING_BATCH_SIZE:-${EEG_BATCH_SIZE:-4}}"
LISTEN_BATCH="${EEG_LISTENING_BATCH_SIZE:-${EEG_BATCH_SIZE:-4}}"
read -r -a GPUS <<< "${EEG_GPUS:-0 1}"
[[ ${#GPUS[@]} -eq 2 ]] || { echo 'Use EEG_GPUS="0 1"' >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$CKPT_ROOT" "$STAGING"
cat > "$RUN_ROOT/metadata.txt" <<META
Order: reading -> listening
Reading datasets: EEGDash (delong, control) + ZuCo (NR)
Pipelines: MEG-XL pretrained and from scratch
Band: 0.1-50 Hz; target 50 Hz; tokenizer BioCodec
GPUs: pretrained=${GPUS[0]}, from_scratch=${GPUS[1]}
META
