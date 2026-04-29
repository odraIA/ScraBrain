#!/usr/bin/env bash
# ==============================================================================
# run_sweep.sh — Búsqueda de hiperparámetros MEG (multi-task, multi-backbone)
# ==============================================================================
#
# Uso:
#   bash run_sweep.sh              # sweep completo
#   bash run_sweep.sh --dry-run    # ver qué se lanzaría sin ejecutar nada
#   bash run_sweep.sh --resume     # reanudar experimentos fallidos (resume_from=latest)
#   bash run_sweep.sh --rerun      # relanzar experimentos desde cero, ignorando .exp_done_*
#   bash run_sweep.sh --rerun-precompute  # recalcular stats aunque exista .precompute_done_*
#   bash run_sweep.sh --detach     # dejar el coordinador corriendo sin depender de la terminal
#   bash run_sweep.sh --ddp        # alias legacy: mantiene train_ddp.py (raw+CWT)
#   bash run_sweep.sh --speech-image        # sweep A–F (imagen TF + ImageNet)
#   bash run_sweep.sh --speech-image --low-freq-bias
#   bash run_sweep.sh --source-projection-path ./libribrain/source_W.npy
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

# ── Rutas base ────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${PROJECT_DIR}/$(basename "${BASH_SOURCE[0]}")"
export PROJECT_DIR

# ── Flags ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
RESUME_FAILED=false
FORCE_RERUN=false
RERUN_PRECOMPUTE=false
SPEECH_IMAGE_SWEEP=false
USE_WANDB=false
LOW_FREQ_BIAS=false
DETACH_REQUESTED=false
SOURCE_RETRAIN_BEST=false
SOURCE_PROJECTION_PATH=""
SOURCE_VARIANT_NAME="source_lcmv"
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  arg="$1"
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --resume)  RESUME_FAILED=true ;;
    --rerun|--force) FORCE_RERUN=true ;;
    --rerun-precompute|--force-precompute) RERUN_PRECOMPUTE=true ;;
    --detach|--daemon|--bg) DETACH_REQUESTED=true; continue ;;
    --ddp)     ;;
    --speech-image) SPEECH_IMAGE_SWEEP=true ;;
    --use-wandb) USE_WANDB=true ;;
    --low-freq-bias) LOW_FREQ_BIAS=true ;;
    --source-retrain-best) SOURCE_RETRAIN_BEST=true ;;
    --source-projection-path=*)
      SOURCE_PROJECTION_PATH="${arg#*=}"
      SOURCE_RETRAIN_BEST=true
      ;;
    --source-projection-path)
      FORWARD_ARGS+=("$arg")
      shift
      if [[ $# -eq 0 ]]; then
        echo "--source-projection-path requiere una ruta" >&2
        exit 2
      fi
      SOURCE_PROJECTION_PATH="$1"
      SOURCE_RETRAIN_BEST=true
      arg="$1"
      ;;
    --source-variant-name=*)
      SOURCE_VARIANT_NAME="${arg#*=}"
      ;;
    --source-variant-name)
      FORWARD_ARGS+=("$arg")
      shift
      if [[ $# -eq 0 ]]; then
        echo "--source-variant-name requiere un nombre" >&2
        exit 2
      fi
      SOURCE_VARIANT_NAME="$1"
      arg="$1"
      ;;
  esac
  FORWARD_ARGS+=("$arg")
  shift
done

if $SOURCE_RETRAIN_BEST && [[ -z "$SOURCE_PROJECTION_PATH" ]]; then
  echo "--source-retrain-best requiere --source-projection-path <matriz>" >&2
  exit 2
fi

# ── Coordinador desacoplado ───────────────────────────────────────────────────
SWEEP_KIND="classic"
if $SPEECH_IMAGE_SWEEP; then
  SWEEP_KIND="speech_image"
fi

mkdir -p "${PROJECT_DIR}/logs"
PID_FILE="${PROJECT_DIR}/.sweep_coordinator_${SWEEP_KIND}.pid"
LATEST_COORDINATOR_LINK="${PROJECT_DIR}/logs/latest_${SWEEP_KIND}_coordinator.log"
LATEST_SWEEP_LINK="${PROJECT_DIR}/logs/latest_${SWEEP_KIND}_sweep.log"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    if $DETACH_REQUESTED && [[ "${SWEEP_COORDINATOR_DETACHED:-0}" != "1" ]]; then
      echo "Ya hay un coordinador '${SWEEP_KIND}' corriendo con PID ${EXISTING_PID}."
      echo "Log: ${LATEST_COORDINATOR_LINK}"
      echo "Para pararlo: kill ${EXISTING_PID}"
      exit 1
    fi
  else
    rm -f "$PID_FILE"
  fi
fi

if $DETACH_REQUESTED && [[ "${SWEEP_COORDINATOR_DETACHED:-0}" != "1" ]]; then
  SWEEP_RUN_TS="$(date +%Y%m%d_%H%M%S)"
  COORDINATOR_LOG="${PROJECT_DIR}/logs/sweep_coordinator_${SWEEP_KIND}_${SWEEP_RUN_TS}.log"

  ln -sfn "$(basename "$COORDINATOR_LOG")" "$LATEST_COORDINATOR_LINK"

  SWEEP_COORDINATOR_DETACHED=1 \
  SWEEP_COORDINATOR_PID_FILE="$PID_FILE" \
  SWEEP_RUN_TS="$SWEEP_RUN_TS" \
  nohup bash "$SCRIPT_PATH" "${FORWARD_ARGS[@]}" \
    > "$COORDINATOR_LOG" 2>&1 < /dev/null &

  CHILD_PID=$!
  echo "$CHILD_PID" > "$PID_FILE"
  sleep 1

  if kill -0 "$CHILD_PID" 2>/dev/null; then
    echo "Sweep '${SWEEP_KIND}' lanzado en background."
    echo "PID: ${CHILD_PID}"
    echo "Log coordinador: ${COORDINATOR_LOG}"
    echo "Log sweep: ${PROJECT_DIR}/logs/sweep_${SWEEP_RUN_TS}.log"
    echo "Symlink log: ${LATEST_COORDINATOR_LINK}"
    echo "Parar: kill ${CHILD_PID}"
    exit 0
  fi

  if wait "$CHILD_PID"; then
    echo "Sweep '${SWEEP_KIND}' arrancó y terminó antes de la comprobación inicial."
    echo "Log coordinador: ${COORDINATOR_LOG}"
    echo "Log sweep: ${PROJECT_DIR}/logs/sweep_${SWEEP_RUN_TS}.log"
    exit 0
  fi

  rm -f "$PID_FILE"
  echo "No se pudo lanzar el coordinador en background. Revisa ${COORDINATOR_LOG}" >&2
  exit 1
fi

# ── Colores ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
log()      { echo -e "[$(date '+%H:%M:%S')] $*" | tee -a "$SWEEP_LOG"; }
log_ok()   { echo -e "${GREEN}[✓]${NC} $*" | tee -a "$SWEEP_LOG"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $*" | tee -a "$SWEEP_LOG"; }
log_err()  { echo -e "${RED}[✗]${NC} $*" | tee -a "$SWEEP_LOG"; }
log_step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$SWEEP_LOG"
             echo -e "${BLUE}  $*${NC}" | tee -a "$SWEEP_LOG"
             echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$SWEEP_LOG"; }

find_resume_checkpoint() {
  local ckpt_dir="$1"
  local latest_link="${ckpt_dir}/checkpoint_latest.pt"

  if [[ -f "$latest_link" ]]; then
    printf '%s\n' "$latest_link"
    return 0
  fi

  local latest_epoch=""
  latest_epoch="$(find "$ckpt_dir" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort | tail -n 1)"
  if [[ -n "$latest_epoch" ]]; then
    printf '%s\n' "$latest_epoch"
    return 0
  fi

  return 1
}

# ── Directorios ───────────────────────────────────────────────────────────────
SWEEP_RUN_TS="${SWEEP_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_LOG="${PROJECT_DIR}/logs/sweep_${SWEEP_RUN_TS}.log"
touch "$SWEEP_LOG"
ln -sfn "$(basename "$SWEEP_LOG")" "$LATEST_SWEEP_LINK"
SWEEP_PLAN="${PROJECT_DIR}/.sweep_plan.json"
export SWEEP_LOG SWEEP_RUN_TS SWEEP_PLAN

plan_set_experiment_status() {
  local exp_name="$1"
  local exp_status="$2"

  python3 - "$SWEEP_PLAN" "$exp_name" "$exp_status" <<'PYEOF'
import json
import sys
from datetime import datetime
from pathlib import Path

plan_path = Path(sys.argv[1])
exp_name = sys.argv[2]
exp_status = sys.argv[3]
now = datetime.now().isoformat()

if not plan_path.exists():
    sys.exit(0)

with plan_path.open(encoding="utf-8") as f:
    data = json.load(f)

experiments = data.get("experiments", [])
normalized = []
found = False
for item in experiments:
    if isinstance(item, str):
        entry = {"name": item}
    elif isinstance(item, dict):
        entry = dict(item)
    else:
        continue

    if entry.get("name") == exp_name:
        entry["status"] = exp_status
        entry["updated_at"] = now
        found = True
    normalized.append(entry)

if not found:
    normalized.append({
        "name": exp_name,
        "status": exp_status,
        "updated_at": now,
    })

data["experiments"] = normalized

with plan_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PYEOF
}

plan_set_precompute_status() {
  local stage_key="$1"
  local task_name="$2"
  local task_status="$3"

  python3 - "$SWEEP_PLAN" "$stage_key" "$task_name" "$task_status" <<'PYEOF'
import json
import sys
from datetime import datetime
from pathlib import Path

plan_path = Path(sys.argv[1])
stage_key = sys.argv[2]
task_name = sys.argv[3]
task_status = sys.argv[4]
now = datetime.now().isoformat()

if not plan_path.exists():
    sys.exit(0)

with plan_path.open(encoding="utf-8") as f:
    data = json.load(f)

precompute = data.setdefault("precompute", {})
stage_items = precompute.get(stage_key, [])
normalized = []
found = False
for item in stage_items:
    if not isinstance(item, dict):
        continue
    entry = dict(item)
    if entry.get("task") == task_name:
        entry["status"] = task_status
        entry["updated_at"] = now
        found = True
    normalized.append(entry)

if not found:
    normalized.append({
        "task": task_name,
        "status": task_status,
        "updated_at": now,
    })

precompute[stage_key] = normalized

with plan_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PYEOF
}

cleanup_pid_file() {
  local pid_file="${SWEEP_COORDINATOR_PID_FILE:-}"
  local recorded_pid

  if [[ -z "$pid_file" ]] || [[ ! -f "$pid_file" ]]; then
    return
  fi

  recorded_pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ "$recorded_pid" == "$$" ]]; then
    rm -f "$pid_file"
  fi
}

trap cleanup_pid_file EXIT

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

to_workspace_path() {
  local path="$1"
  while [[ "$path" == ./* ]]; do
    path="${path#./}"
  done
  if [[ "$path" == /workspace/* ]]; then
    printf '%s\n' "$path"
  elif [[ "$path" == "$PROJECT_DIR"/* ]]; then
    printf '/workspace/%s\n' "${path#"$PROJECT_DIR"/}"
  elif [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '/workspace/%s\n' "$path"
  fi
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
now = datetime.now().isoformat()

payload = {
    "generated_at": now,
    "mode": "speech_image",
    "run_ts": os.environ.get("SWEEP_RUN_TS"),
    "sweep_log": f"logs/{os.path.basename(os.environ['SWEEP_LOG'])}",
    "experiments": [
        {
            "name": exp,
            "status": "pending",
            "updated_at": now,
        }
        for exp in experiments
    ],
    "precompute": {
        "stats": [],
        "images": [],
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

    if [[ -f "$DONE_SENTINEL" ]] && ! $RESUME_FAILED && ! $FORCE_RERUN; then
      plan_set_experiment_status "$EXP" "skipped"
      log_ok "  Ya completado (sentinel existente). Saltando."
      SKIPPED+=("$EXP")
      continue
    fi

    if [[ -f "$DONE_SENTINEL" ]] && $FORCE_RERUN && ! $DRY_RUN; then
      rm -f "$DONE_SENTINEL"
      log_warn "  Relanzando desde cero: sentinel anterior eliminado."
    elif [[ -f "$DONE_SENTINEL" ]] && $FORCE_RERUN; then
      log_warn "  [DRY-RUN] Se ignoraría/eliminaría sentinel anterior para relanzar."
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
    plan_set_experiment_status "$EXP" "running"
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
      plan_set_experiment_status "$EXP" "done"
      log_ok "  Completado en ${ELAPSED_MIN}min → ${JOB_LOG}"
      PASSED+=("$EXP")
    else
      plan_set_experiment_status "$EXP" "failed"
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
BATCH_SIZE=128       # Por GPU — batch global = 256 × 2 GPUs = 512
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
now = datetime.now().isoformat()

payload = {
    "generated_at": now,
    "mode": "classic",
    "run_ts": os.environ.get("SWEEP_RUN_TS"),
    "sweep_log": f"logs/{os.path.basename(os.environ['SWEEP_LOG'])}",
    "experiments": [
        {
            "name": f"{t}__{b}__{s}",
            "status": "pending",
            "updated_at": now,
        }
        for t, b, s in itertools.product(tasks, backbones, strategies)
    ],
    "precompute": {
        "stats": [
            {
                "task": task,
                "status": "pending",
                "updated_at": now,
            }
            for task in tasks
        ],
        "images": [],
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

  if [[ -f "$STATS_SENTINEL" ]] && ! $RERUN_PRECOMPUTE; then
    plan_set_precompute_status "stats" "$TASK" "skipped"
    log_ok "Stats para '${TASK}' ya calculadas (sentinel: ${STATS_SENTINEL})"
    continue
  fi

  if [[ -f "$STATS_SENTINEL" ]] && $RERUN_PRECOMPUTE && ! $DRY_RUN; then
    rm -f "$STATS_SENTINEL"
    log_warn "Recalculando stats para '${TASK}': sentinel anterior eliminado."
  elif [[ -f "$STATS_SENTINEL" ]] && $RERUN_PRECOMPUTE; then
    log_warn "[DRY-RUN] Se ignoraría/eliminaría sentinel de stats para '${TASK}'."
  fi

  log "Lanzando precompute_stats para task='${TASK}'..."

  if $DRY_RUN; then
    log "  [DRY-RUN] docker compose run --rm precompute_stats ..."
    continue
  fi

  # Lanzar precompute desacoplado de la sesión SSH y esperar a que termine
  plan_set_precompute_status "stats" "$TASK" "running"
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
    plan_set_precompute_status "stats" "$TASK" "done"
    log_ok "Precompute '${TASK}' completado."
  else
    plan_set_precompute_status "stats" "$TASK" "failed"
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
  if [[ -f "$DONE_SENTINEL" ]] && ! $RESUME_FAILED && ! $FORCE_RERUN; then
    plan_set_experiment_status "$EXP" "skipped"
    log_ok "  Ya completado (sentinel existente). Saltando."
    SKIPPED+=("$EXP")
    continue
  fi

  if [[ -f "$DONE_SENTINEL" ]] && $FORCE_RERUN && ! $DRY_RUN; then
    rm -f "$DONE_SENTINEL"
    log_warn "  Relanzando desde cero: sentinel anterior eliminado."
  elif [[ -f "$DONE_SENTINEL" ]] && $FORCE_RERUN; then
    log_warn "  [DRY-RUN] Se ignoraría/eliminaría sentinel anterior para relanzar."
  fi

  # ── Determinar resume_from ──────────────────────────────────────────────────
  RESUME_FROM="none"
  if $RESUME_FAILED && ! $FORCE_RERUN; then
    # Si hay checkpoint previo para este experimento, reanudar desde él.
    # Fallback: si el proceso cayó tras guardar el .pt pero antes del symlink
    # "latest", usar el checkpoint_epoch_*.pt más reciente.
    CKPT_LOCAL_DIR="${PROJECT_DIR}/checkpoints/${EXP}"
    if RESUME_CKPT="$(find_resume_checkpoint "$CKPT_LOCAL_DIR")"; then
      if [[ "$RESUME_CKPT" == "${CKPT_LOCAL_DIR}/checkpoint_latest.pt" ]]; then
        RESUME_FROM="latest"
      else
        RESUME_FROM="$RESUME_CKPT"
      fi
      log_warn "  Reanudando desde checkpoint: ${RESUME_CKPT}"
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
  plan_set_experiment_status "$EXP" "running"

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
    plan_set_experiment_status "$EXP" "done"
    log_ok "  Completado en ${ELAPSED_MIN}min → ${JOB_LOG}"
    PASSED+=("$EXP")
  else
    plan_set_experiment_status "$EXP" "failed"
    log_err "  FALLIDO (exit ${EXIT_CODE}) en ${ELAPSED_MIN}min → ${JOB_LOG}"
    FAILED+=("$EXP")
    # Continuar con el siguiente experimento (no abortar el sweep)
  fi

done  # STRATEGY
done  # BACKBONE
done  # TASK

# ==============================================================================
# PASO 3: REENTRENAR EL MEJOR EXPERIMENTO EN ESPACIO FUENTE/ROI
# ==============================================================================
if $SOURCE_RETRAIN_BEST; then
  log_step "PASO 3: Reentrenando mejor experimento con proyección fuente antes de CWT"

  SOURCE_PROJECTION_CONTAINER_PATH="$(to_workspace_path "$SOURCE_PROJECTION_PATH")"
  export SOURCE_VARIANT_NAME

  BEST_SPEC="$(python3 - <<'PYEOF'
import glob
import json
import os
import sys
from pathlib import Path

results_base = Path(os.environ.get("PROJECT_DIR", ".")) / "results"
allowed_tasks = {x.strip() for x in os.environ.get("TASKS_CSV", "").split(",") if x.strip()}
allowed_backbones = {x.strip() for x in os.environ.get("BACKBONES_CSV", "").split(",") if x.strip()}
allowed_strategies = {x.strip() for x in os.environ.get("STRATEGIES_CSV", "").split(",") if x.strip()}
rows = []
for path in glob.glob(str(results_base / "*" / "final_results.json")):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        continue
    if d.get("representation", "sensor") != "sensor":
        continue
    task = d.get("task")
    backbone = d.get("backbone")
    strategy = d.get("strategy")
    f1 = d.get("test_f1_macro")
    if allowed_tasks and task not in allowed_tasks:
        continue
    if allowed_backbones and backbone not in allowed_backbones:
        continue
    if allowed_strategies and strategy not in allowed_strategies:
        continue
    if task and backbone and strategy and isinstance(f1, (int, float)):
        rows.append((float(f1), task, backbone, strategy))

if not rows:
    sys.exit(1)

rows.sort(reverse=True)
f1, task, backbone, strategy = rows[0]
print(f"{task}\t{backbone}\t{strategy}\t{f1:.8f}")
PYEOF
  )" || BEST_SPEC=""

  if [[ -z "$BEST_SPEC" ]]; then
    log_warn "No se encontró un final_results.json válido para elegir el mejor experimento. Saltando fuente."
  else
    IFS=$'\t' read -r BEST_TASK BEST_BACKBONE BEST_STRATEGY BEST_F1 <<< "$BEST_SPEC"
    SOURCE_EXP="${BEST_TASK}__${BEST_BACKBONE}__${BEST_STRATEGY}__${SOURCE_VARIANT_NAME}"
    SOURCE_JOB_LOG="${PROJECT_DIR}/logs/${SOURCE_EXP}.log"
    SOURCE_OUTPUT_DIR="/workspace/results/${SOURCE_EXP}"
    SOURCE_CKPT_DIR="/workspace/checkpoints/${SOURCE_EXP}"
    SOURCE_SENTINEL="${PROJECT_DIR}/.exp_done_${SOURCE_EXP}"

    log "Mejor experimento sensor: ${BEST_TASK} | ${BEST_BACKBONE} | ${BEST_STRATEGY} (F1=${BEST_F1})"
    log "Proyección fuente: ${SOURCE_PROJECTION_PATH} -> ${SOURCE_PROJECTION_CONTAINER_PATH}"

    if [[ -f "$SOURCE_SENTINEL" ]] && ! $RESUME_FAILED && ! $FORCE_RERUN; then
      plan_set_experiment_status "$SOURCE_EXP" "skipped"
      log_ok "  Fuente ya completado (sentinel existente). Saltando."
      SKIPPED+=("$SOURCE_EXP")
    elif $DRY_RUN; then
      log "  [DRY-RUN] docker compose run ${DOCKER_COMPOSE_RUN_FLAGS_STR} --rm meg_training_job train_ddp.py \\"
      log "    --task ${BEST_TASK} --backbone ${BEST_BACKBONE} --strategy ${BEST_STRATEGY} \\"
      log "    --source_projection_path ${SOURCE_PROJECTION_CONTAINER_PATH} --source_variant_name ${SOURCE_VARIANT_NAME} \\"
      log "    --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} \\"
      log "    --output_dir ${SOURCE_OUTPUT_DIR} --checkpoint_dir ${SOURCE_CKPT_DIR}"
      PASSED+=("$SOURCE_EXP [dry-run]")
    else
      if [[ -f "$SOURCE_SENTINEL" ]] && $FORCE_RERUN; then
        rm -f "$SOURCE_SENTINEL"
        log_warn "  Relanzando fuente desde cero: sentinel anterior eliminado."
      fi

      START_TS=$(date +%s)
      plan_set_experiment_status "$SOURCE_EXP" "running"
      SOURCE_CONTAINER_ID=$(compose_run_detached \
        meg_training_job \
        train_ddp.py \
          --task          "${BEST_TASK}" \
          --backbone      "${BEST_BACKBONE}" \
          --strategy      "${BEST_STRATEGY}" \
          --n_epochs      "${N_EPOCHS}" \
          --batch_size    "${BATCH_SIZE}" \
          --num_workers   "${NUM_WORKERS}" \
          --data_path     "${DATA_PATH}" \
          --output_dir    "${SOURCE_OUTPUT_DIR}" \
          --checkpoint_dir "${SOURCE_CKPT_DIR}" \
          --checkpoint_every "${CHECKPOINT_EVERY}" \
          --resume_from   "none" \
          --source_projection_path "${SOURCE_PROJECTION_CONTAINER_PATH}" \
          --source_variant_name "${SOURCE_VARIANT_NAME}")

      log "  Contenedor fuente: ${SOURCE_CONTAINER_ID:0:12}"
      docker logs -f "$SOURCE_CONTAINER_ID" 2>&1 | tee "$SOURCE_JOB_LOG" &
      LOGS_PID=$!

      EXIT_CODE=$(docker wait "$SOURCE_CONTAINER_ID")
      kill $LOGS_PID 2>/dev/null || true
      docker rm "$SOURCE_CONTAINER_ID" 2>/dev/null || true
      ELAPSED=$(( $(date +%s) - START_TS ))
      ELAPSED_MIN=$(( ELAPSED / 60 ))

      if [[ $EXIT_CODE -eq 0 ]]; then
        touch "$SOURCE_SENTINEL"
        plan_set_experiment_status "$SOURCE_EXP" "done"
        log_ok "  Fuente completado en ${ELAPSED_MIN}min → ${SOURCE_JOB_LOG}"
        PASSED+=("$SOURCE_EXP")
      else
        plan_set_experiment_status "$SOURCE_EXP" "failed"
        log_err "  Fuente FALLIDO (exit ${EXIT_CODE}) en ${ELAPSED_MIN}min → ${SOURCE_JOB_LOG}"
        FAILED+=("$SOURCE_EXP")
      fi
    fi
  fi
fi

# ==============================================================================
# PASO 4: RESUMEN FINAL
# ==============================================================================
log_step "RESUMEN DEL SWEEP"

SUMMARY_TOTAL=$TOTAL
if $SOURCE_RETRAIN_BEST; then
  SUMMARY_TOTAL=$(( SUMMARY_TOTAL + 1 ))
fi

log "Completados OK (${#PASSED[@]}/${SUMMARY_TOTAL}):"
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
print(f"\n{'Experimento':<58} {'Repr':<12} {'Task':<10} {'Backbone':<18} {'Strategy':<12} {'Test F1':>8} {'Bal Acc':>8} {'AUROC':>8}")
print("─" * 142)
for r in rows:
    repr_name = r.get("representation", "sensor")
    exp = f"{r.get('task','?')}__{r.get('backbone','?')}__{r.get('strategy','?')}"
    if repr_name != "sensor":
        exp = f"{exp}__{repr_name}"
    print(
        f"  {exp:<56} "
        f"{repr_name:<12} "
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
