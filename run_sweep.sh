#!/usr/bin/env bash
# ==============================================================================
# run_sweep.sh — Búsqueda de hiperparámetros MEG (multi-task, multi-backbone)
# ==============================================================================
#
# Uso:
#   bash run_sweep.sh              # sweep completo
#   bash run_sweep.sh --dry-run    # ver qué se lanzaría sin ejecutar nada
#   bash run_sweep.sh --resume     # reanudar experimentos fallidos (resume_from=latest)
#   bash run_sweep.sh --ddp        # alias legacy: mantiene train_ddp.py (raw+CWT)
#   bash run_sweep.sh --speech-image        # sweep A–F (imagen TF + ImageNet)
#   bash run_sweep.sh --speech-image --low-freq-bias
#
# Prerrequisitos:
#   - docker compose up --build  (al menos una vez para construir la imagen)
#   - precompute_stats corrido para 'phoneme' (ver abajo, lo hace automáticamente)
#   - train_ddp.py parcheado para speech (este script lo hace automáticamente)
#
# Resultados:
#   logs/
#     sweep_<timestamp>.log                ← progreso global del sweep
#     <task>__<backbone>__<strategy>.log   ← stdout+stderr de cada job
#   results/<task>__<backbone>__<strategy>/
#     final_results.json                   ← métricas finales de test
#     tensorboard/                         ← curvas de entrenamiento
#   checkpoints/<task>__<backbone>__<strategy>/
#     best_model.pt                        ← mejor checkpoint por val F1
#     checkpoint_epoch_XXXX.pt
# ==============================================================================

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
RESUME_FAILED=false
SPEECH_IMAGE_SWEEP=false
USE_WANDB=false
LOW_FREQ_BIAS=false
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --resume)  RESUME_FAILED=true ;;
    --ddp)     ;;
    --speech-image) SPEECH_IMAGE_SWEEP=true ;;
    --use-wandb) USE_WANDB=true ;;
    --low-freq-bias) LOW_FREQ_BIAS=true ;;
  esac
done

# ── Colores ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
log()      { echo -e "[$(date '+%H:%M:%S')] $*" | tee -a "$SWEEP_LOG"; }
log_ok()   { echo -e "${GREEN}[✓]${NC} $*" | tee -a "$SWEEP_LOG"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $*" | tee -a "$SWEEP_LOG"; }
log_err()  { echo -e "${RED}[✗]${NC} $*" | tee -a "$SWEEP_LOG"; }
log_step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$SWEEP_LOG"
             echo -e "${BLUE}  $*${NC}" | tee -a "$SWEEP_LOG"
             echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$SWEEP_LOG"; }

# ── Directorios ───────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR
mkdir -p "${PROJECT_DIR}/logs"
SWEEP_LOG="${PROJECT_DIR}/logs/sweep_$(date +%Y%m%d_%H%M%S).log"
touch "$SWEEP_LOG"
SWEEP_PLAN="${PROJECT_DIR}/.sweep_plan.json"

# ── Compatibilidad docker compose run ─────────────────────────────────────────
DOCKER_COMPOSE_RUN_FLAGS=(-d --no-deps)
if command -v docker >/dev/null 2>&1; then
  if docker compose run --help 2>/dev/null | grep -q -- '--init'; then
    DOCKER_COMPOSE_RUN_FLAGS=(-d --init --no-deps)
  else
    log_warn "docker compose run no soporta --init; se ejecutará sin ese flag."
  fi
fi
DOCKER_COMPOSE_RUN_FLAGS_STR="${DOCKER_COMPOSE_RUN_FLAGS[*]}"

compose_run_detached() {
  docker compose run "${DOCKER_COMPOSE_RUN_FLAGS[@]}" "$@"
}

# ==============================================================================
# MODO ALTERNATIVO: SWEEP DE SPEECH IMAGE EXPERIMENTS (A–F)
# ==============================================================================
if $SPEECH_IMAGE_SWEEP; then
  echo "speech_image" > "${PROJECT_DIR}/.sweep_mode"
  export SWEEP_PLAN
  log_step "MODO SPEECH-IMAGE: lanzando experimentos A–F"

  SPEECH_EXPERIMENTS=(
    "baseline_image_resnet18"
    "baseline_image_vittiny"
    "ablation_projection"
    "ablation_finetuning"
    "ablation_window_length"
    "ablation_augmentations"
  )
  if $LOW_FREQ_BIAS; then
    SPEECH_EXPERIMENTS+=("low_freq_bias_variant")
  fi
  export SWEEP_PLAN
  export SPEECH_EXPERIMENTS_CSV
  SPEECH_EXPERIMENTS_CSV="$(IFS=,; echo "${SPEECH_EXPERIMENTS[*]}")"

  python3 - <<'PYEOF'
import json
import os
from datetime import datetime

raw_experiments = [
    exp.strip()
    for exp in os.environ.get("SPEECH_EXPERIMENTS_CSV", "").split(",")
    if exp.strip()
]
experiments = [f"speech_image__{exp}" for exp in raw_experiments]

payload = {
    "generated_at": datetime.now().isoformat(),
    "mode": "speech_image",
    "experiments": experiments,
    "precompute": {
        "stats_tasks": [],
        "images_tasks": [],
    },
}

with open(os.environ["SWEEP_PLAN"], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PYEOF

  TOTAL=${#SPEECH_EXPERIMENTS[@]}
  PASSED=()
  FAILED=()
  SKIPPED=()
  EXP_NUM=0

  for EXP_ID in "${SPEECH_EXPERIMENTS[@]}"; do
    EXP_NUM=$(( EXP_NUM + 1 ))
    EXP="speech_image__${EXP_ID}"
    JOB_LOG="${PROJECT_DIR}/logs/${EXP}.log"
    OUT_DIR="/workspace/results/${EXP}"
    DONE_SENTINEL="${PROJECT_DIR}/.exp_done_${EXP}"

    log ""
    log "━━━ [${EXP_NUM}/${TOTAL}] ${EXP} ━━━"

    if [[ -f "$DONE_SENTINEL" ]] && ! $RESUME_FAILED; then
      log_ok "  Ya completado (sentinel existente). Saltando."
      SKIPPED+=("$EXP")
      continue
    fi

    TF_VARIANT="full_band_tf"
    if [[ "$EXP_ID" == "low_freq_bias_variant" ]]; then
      TF_VARIANT="low_freq_biased_tf"
    fi

    WB_FLAGS=""
    if $USE_WANDB; then
      WB_FLAGS="--use_wandb"
    fi

    if $DRY_RUN; then
      log "  [DRY-RUN] docker compose run ${DOCKER_COMPOSE_RUN_FLAGS_STR} --rm --entrypoint python meg_training_job \\"
      log "    /workspace/run_speech_image_experiments.py \\"
      log "    --experiment ${EXP_ID} --data_path /workspace/libribrain_data \\"
      log "    --output_dir ${OUT_DIR} --epochs 20 --stage1_epochs 6 \\"
      log "    --batch_size 32 --num_workers 4 --seeds 42,43,44 --tf_variant ${TF_VARIANT} ${WB_FLAGS}"
      PASSED+=("$EXP [dry-run]")
      continue
    fi

    START_TS=$(date +%s)
    CONTAINER_ID=$(compose_run_detached \
      --entrypoint python \
      meg_training_job \
      /workspace/run_speech_image_experiments.py \
        --experiment "${EXP_ID}" \
        --data_path /workspace/libribrain_data \
        --output_dir "${OUT_DIR}" \
        --epochs 20 \
        --stage1_epochs 6 \
        --batch_size 32 \
        --num_workers 4 \
        --seeds 42,43,44 \
        --tf_variant "${TF_VARIANT}" \
        ${WB_FLAGS})

    log "  Contenedor: ${CONTAINER_ID:0:12}"
    docker logs -f "$CONTAINER_ID" 2>&1 | tee "$JOB_LOG" &
    LOGS_PID=$!

    EXIT_CODE=$(docker wait "$CONTAINER_ID")
    kill $LOGS_PID 2>/dev/null || true
    docker rm "$CONTAINER_ID" 2>/dev/null || true
    ELAPSED=$(( $(date +%s) - START_TS ))
    ELAPSED_MIN=$(( ELAPSED / 60 ))

    if [[ "$EXIT_CODE" -eq 0 ]]; then
      touch "$DONE_SENTINEL"
      log_ok "  Completado en ${ELAPSED_MIN}min → ${JOB_LOG}"
      PASSED+=("$EXP")
    else
      log_err "  FALLIDO (exit ${EXIT_CODE}) en ${ELAPSED_MIN}min → ${JOB_LOG}"
      FAILED+=("$EXP")
    fi
  done

  log_step "RESUMEN DEL SWEEP (SPEECH-IMAGE)"
  log "Completados OK (${#PASSED[@]}/${TOTAL}):"
  for e in "${PASSED[@]:-}"; do log_ok "  ${e}"; done
  if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    log "Saltados (ya existían) (${#SKIPPED[@]}):"
    for e in "${SKIPPED[@]}"; do log "  ⏭  ${e}"; done
  fi
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    log "Fallidos (${#FAILED[@]}):"
    for e in "${FAILED[@]}"; do log_err "  ${e}"; done
  fi

  log_step "RESULTADOS (SPEECH-IMAGE)"
  python3 - << 'PYEOF'
import json, os, glob, sys
from pathlib import Path

results_base = Path(os.environ.get("PROJECT_DIR", ".")) / "results"
jsons = sorted(glob.glob(str(results_base / "speech_image__*" / "final_results.json")))
if not jsons:
    print("  No hay resultados disponibles todavía.")
    sys.exit(0)

rows = []
for path in jsons:
    with open(path) as f:
        d = json.load(f)
    exp = Path(path).parent.name
    rows.append((exp, d))

rows.sort(key=lambda x: x[1].get("test_f1_macro", 0), reverse=True)
print(f"\n{'Experimento':<38} {'Backbone':<10} {'Proj':<26} {'FT':<12} {'F1':>7} {'BalAcc':>7} {'AUROC':>7}")
print("─" * 122)
for exp, d in rows:
    print(
        f"  {exp:<36} "
        f"{str(d.get('backbone','?')):<10} "
        f"{str(d.get('projection_type','?')):<26} "
        f"{str(d.get('fine_tuning_type','?')):<12} "
        f"{float(d.get('test_f1_macro',0)):.4f} "
        f"{float(d.get('test_balanced_acc',0)):.4f} "
        f"{float(d.get('test_auroc', float('nan'))):.4f}"
    )
print()
PYEOF

  log_ok "Sweep speech-image finalizado."
  exit 0
fi

echo "classic" > "${PROJECT_DIR}/.sweep_mode"

# ==============================================================================
# ESPACIO DE BÚSQUEDA
# Editar aquí para añadir/quitar experimentos.
# ==============================================================================
TASKS=("speech") # "phoneme"  # Por ahora solo speech, para validar el flujo completo. Phoneme se mantiene como sanity check (precompute + entrenamiento rápido).
BACKBONES=("resnet18" "efficientnet_b0" "vit_tiny")
STRATEGIES=("frozen" "partial_ft")

# Hiperparámetros comunes (se pueden hacer arrays para grid search)
N_EPOCHS=50
BATCH_SIZE=256       # Por GPU — batch global = 256 × 2 GPUs = 512
NUM_WORKERS=4
CHECKPOINT_EVERY=5   # Guardar checkpoint cada 5 epochs (no cada 1, para no llenar disco)
DATA_PATH="/workspace/libribrain_data"
export SWEEP_PLAN
export TASKS_CSV BACKBONES_CSV STRATEGIES_CSV
TASKS_CSV="$(IFS=,; echo "${TASKS[*]}")"
BACKBONES_CSV="$(IFS=,; echo "${BACKBONES[*]}")"
STRATEGIES_CSV="$(IFS=,; echo "${STRATEGIES[*]}")"
python3 - <<'PYEOF'
import itertools
import json
import os
from datetime import datetime

tasks = [x.strip() for x in os.environ.get("TASKS_CSV", "").split(",") if x.strip()]
backbones = [x.strip() for x in os.environ.get("BACKBONES_CSV", "").split(",") if x.strip()]
strategies = [x.strip() for x in os.environ.get("STRATEGIES_CSV", "").split(",") if x.strip()]

payload = {
    "generated_at": datetime.now().isoformat(),
    "mode": "classic",
    "experiments": [f"{t}__{b}__{s}" for t, b, s in itertools.product(tasks, backbones, strategies)],
    "precompute": {
        "stats_tasks": tasks,
        "images_tasks": [],
    },
}

with open(os.environ["SWEEP_PLAN"], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PYEOF

# ── Modo de entrenamiento ─────────────────────────────────────────────────────
# Se mantiene train_ddp.py (DDP raw+CWT, flujo actual).

# ==============================================================================
# PASO 0: PARCHE DE train_ddp.py PARA SOPORTE DE SPEECH
# ==============================================================================
# train_ddp.py original tiene el cálculo de class_weights hardcodeado para
# phoneme. El parche lo hace task-aware (phoneme Y speech).

log_step "PASO 0: Verificando/parcheando train_ddp.py para soporte de 'speech'"

TRAIN_DDP="${PROJECT_DIR}/train_ddp.py"

if grep -q '_extract_train_labels_fast' "$TRAIN_DDP" 2>/dev/null; then
  log_ok "train_ddp.py ya incluye soporte robusto para speech/phoneme. Sin parche dinámico."
else
  log_warn "No se detecta la versión nueva de train_ddp.py. Se continúa sin autoparche (legacy)."
fi

# ==============================================================================
# PASO 1: PRECOMPUTE STATS PARA AMBAS TAREAS
# ==============================================================================
log_step "PASO 1: Precompute stats de normalización H5"

for TASK in "${TASKS[@]}"; do
  # Detectar si ya está cacheado (heurística: si existe el directorio de datos
  # y no hay ningún lock reciente, asumimos que precompute ya corrió)
  STATS_SENTINEL="${PROJECT_DIR}/.precompute_done_${TASK}"

  if [[ -f "$STATS_SENTINEL" ]]; then
    log_ok "Stats para '${TASK}' ya calculadas (sentinel: ${STATS_SENTINEL})"
    continue
  fi

  log "Lanzando precompute_stats para task='${TASK}'..."

  if $DRY_RUN; then
    log "  [DRY-RUN] docker compose run --rm precompute_stats ..."
    continue
  fi

  # Lanzar precompute desacoplado de la sesión SSH y esperar a que termine
  PRECOMPUTE_CID=$(compose_run_detached \
    -e "TASK_OVERRIDE=${TASK}" \
    precompute_stats \
    /workspace/precompute_stats.py \
      --data_path "${DATA_PATH}" \
      --task "${TASK}")

  # Volcar logs en tiempo real mientras esperamos
  docker logs -f "$PRECOMPUTE_CID" 2>&1 | tee "${PROJECT_DIR}/logs/precompute_${TASK}.log" &
  LOGS_PID=$!

  # Esperar a que el contenedor termine y recoger su exit code
  EXIT_CODE=$(docker wait "$PRECOMPUTE_CID")
  kill $LOGS_PID 2>/dev/null || true
  docker rm "$PRECOMPUTE_CID" 2>/dev/null || true

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    touch "$STATS_SENTINEL"
    log_ok "Precompute '${TASK}' completado."
  else
    log_err "Precompute '${TASK}' falló (exit ${EXIT_CODE}). Ver logs/precompute_${TASK}.log"
    exit 1
  fi
done

# ==============================================================================
# PASO 2: SWEEP DE ENTRENAMIENTO
# ==============================================================================
log_step "PASO 2: Iniciando sweep de entrenamiento"

# Calcular total de experimentos
TOTAL=$(( ${#TASKS[@]} * ${#BACKBONES[@]} * ${#STRATEGIES[@]} ))
log "Espacio de búsqueda:"
log "  Tareas:      ${TASKS[*]}"
log "  Backbones:   ${BACKBONES[*]}"
log "  Estrategias: ${STRATEGIES[*]}"
log "  Total:       ${TOTAL} experimentos"
log "  Epochs/exp:  ${N_EPOCHS}"
log "  Modo train:  DDP raw+CWT (train_ddp.py)"
log ""

FAILED=()
PASSED=()
SKIPPED=()
EXP_NUM=0

for TASK in "${TASKS[@]}"; do
for BACKBONE in "${BACKBONES[@]}"; do
for STRATEGY in "${STRATEGIES[@]}"; do

  EXP_NUM=$(( EXP_NUM + 1 ))
  EXP="${TASK}__${BACKBONE}__${STRATEGY}"
  JOB_LOG="${PROJECT_DIR}/logs/${EXP}.log"
  OUTPUT_DIR="/workspace/results/${EXP}"
  CKPT_DIR="/workspace/checkpoints/${EXP}"
  DONE_SENTINEL="${PROJECT_DIR}/.exp_done_${EXP}"

  log ""
  log "━━━ [${EXP_NUM}/${TOTAL}] ${EXP} ━━━"

  # ── Saltar experimentos ya completados ──────────────────────────────────────
  if [[ -f "$DONE_SENTINEL" ]] && ! $RESUME_FAILED; then
    log_ok "  Ya completado (sentinel existente). Saltando."
    SKIPPED+=("$EXP")
    continue
  fi

  # ── Determinar resume_from ──────────────────────────────────────────────────
  RESUME_FROM="none"
  if $RESUME_FAILED; then
    # Si hay checkpoint previo para este experimento, reanudar desde él
    CKPT_LOCAL="${PROJECT_DIR}/checkpoints/${EXP}/checkpoint_latest.pt"
    if [[ -f "$CKPT_LOCAL" ]]; then
      RESUME_FROM="latest"
      log_warn "  Reanudando desde checkpoint: ${CKPT_LOCAL}"
    fi
  fi

  if $DRY_RUN; then
    log "  [DRY-RUN] docker compose run ${DOCKER_COMPOSE_RUN_FLAGS_STR} --rm meg_training_job train_ddp.py \\"
    log "    --task ${TASK} --backbone ${BACKBONE} --strategy ${STRATEGY} \\"
    log "    --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} \\"
    log "    --output_dir ${OUTPUT_DIR} --checkpoint_dir ${CKPT_DIR} \\"
    log "    --checkpoint_every ${CHECKPOINT_EVERY} --resume_from ${RESUME_FROM}"
    PASSED+=("$EXP [dry-run]")
    continue
  fi

  START_TS=$(date +%s)

  # ── Lanzar job desacoplado de la sesión SSH ────────────────────────────────
  CONTAINER_ID=$(compose_run_detached \
    meg_training_job \
    train_ddp.py \
      --task          "${TASK}" \
      --backbone      "${BACKBONE}" \
      --strategy      "${STRATEGY}" \
      --n_epochs      "${N_EPOCHS}" \
      --batch_size    "${BATCH_SIZE}" \
      --num_workers   "${NUM_WORKERS}" \
      --data_path     "${DATA_PATH}" \
      --output_dir    "${OUTPUT_DIR}" \
      --checkpoint_dir "${CKPT_DIR}" \
      --checkpoint_every "${CHECKPOINT_EVERY}" \
      --resume_from   "${RESUME_FROM}")

  log "  Contenedor: ${CONTAINER_ID:0:12}"

  # Volcar logs en tiempo real al archivo (proceso en background)
  docker logs -f "$CONTAINER_ID" 2>&1 | tee "$JOB_LOG" &
  LOGS_PID=$!

  # Esperar a que el contenedor termine y recoger su exit code real
  EXIT_CODE=$(docker wait "$CONTAINER_ID")
  kill $LOGS_PID 2>/dev/null || true
  docker rm "$CONTAINER_ID" 2>/dev/null || true
  ELAPSED=$(( $(date +%s) - START_TS ))
  ELAPSED_MIN=$(( ELAPSED / 60 ))

  if [[ $EXIT_CODE -eq 0 ]]; then
    touch "$DONE_SENTINEL"
    log_ok "  Completado en ${ELAPSED_MIN}min → ${JOB_LOG}"
    PASSED+=("$EXP")
  else
    log_err "  FALLIDO (exit ${EXIT_CODE}) en ${ELAPSED_MIN}min → ${JOB_LOG}"
    FAILED+=("$EXP")
    # Continuar con el siguiente experimento (no abortar el sweep)
  fi

done  # STRATEGY
done  # BACKBONE
done  # TASK

# ==============================================================================
# PASO 4: RESUMEN FINAL
# ==============================================================================
log_step "RESUMEN DEL SWEEP"

log "Completados OK (${#PASSED[@]}/${TOTAL}):"
for e in "${PASSED[@]:-}"; do log_ok "  ${e}"; done

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  log "Saltados (ya existían) (${#SKIPPED[@]}):"
  for e in "${SKIPPED[@]}"; do log "  ⏭  ${e}"; done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "Fallidos (${#FAILED[@]}):"
  for e in "${FAILED[@]}"; do log_err "  ${e}"; done
  log ""
  log_warn "Para reanudar los fallidos:  bash run_sweep.sh --resume"
fi

log ""
log "Log completo del sweep: ${SWEEP_LOG}"
log ""

# ==============================================================================
# PASO 5: TABLA COMPARATIVA DE RESULTADOS
# ==============================================================================
log_step "RESULTADOS (final_results.json por experimento)"

python3 - << 'PYEOF'
import json, os, glob, sys
from pathlib import Path

results_base = Path(os.environ.get("PROJECT_DIR", ".")) / "results"
jsons = sorted(glob.glob(str(results_base / "*" / "final_results.json")))

if not jsons:
    print("  No hay resultados disponibles todavía.")
    sys.exit(0)

rows = []
for path in jsons:
    with open(path) as f:
        d = json.load(f)
    rows.append(d)

rows.sort(key=lambda r: r.get("test_f1_macro", 0), reverse=True)

# Cabecera
print(f"\n{'Experimento':<45} {'Task':<10} {'Backbone':<18} {'Strategy':<12} {'Test F1':>8} {'Bal Acc':>8} {'AUROC':>8}")
print("─" * 120)
for r in rows:
    exp = f"{r.get('task','?')}__{r.get('backbone','?')}__{r.get('strategy','?')}"
    print(
        f"  {exp:<43} "
        f"{r.get('task','?'):<10} "
        f"{r.get('backbone','?'):<18} "
        f"{r.get('strategy','?'):<12} "
        f"{r.get('test_f1_macro', 0):.4f}   "
        f"{r.get('test_balanced_acc', 0):.4f}   "
        f"{r.get('test_auroc', float('nan')):.4f}"
    )
print()
PYEOF

log ""
log_ok "Sweep finalizado."
