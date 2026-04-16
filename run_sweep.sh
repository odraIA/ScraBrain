#!/usr/bin/env bash
# ==============================================================================
# run_sweep.sh — Búsqueda de hiperparámetros MEG (multi-task, multi-backbone)
# ==============================================================================
#
# Uso:
#   bash run_sweep.sh              # sweep completo
#   bash run_sweep.sh --dry-run    # ver qué se lanzaría sin ejecutar nada
#   bash run_sweep.sh --resume     # reanudar experimentos fallidos (resume_from=latest)
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
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --resume)  RESUME_FAILED=true ;;
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
mkdir -p "${PROJECT_DIR}/logs"
SWEEP_LOG="${PROJECT_DIR}/logs/sweep_$(date +%Y%m%d_%H%M%S).log"
touch "$SWEEP_LOG"

# ==============================================================================
# ESPACIO DE BÚSQUEDA
# Editar aquí para añadir/quitar experimentos.
# ==============================================================================
TASKS=("phoneme" "speech")
BACKBONES=("resnet18" "efficientnet_b0" "vit_tiny")
STRATEGIES=("frozen" "partial_ft")

# Hiperparámetros comunes (se pueden hacer arrays para grid search)
N_EPOCHS=50
BATCH_SIZE=256       # Por GPU — batch global = 256 × 2 GPUs = 512
NUM_WORKERS=4
CHECKPOINT_EVERY=5   # Guardar checkpoint cada 5 epochs (no cada 1, para no llenar disco)
DATA_PATH="/workspace/libribrain_data"

# ==============================================================================
# PASO 0: PARCHE DE train_ddp.py PARA SOPORTE DE SPEECH
# ==============================================================================
# train_ddp.py original tiene el cálculo de class_weights hardcodeado para
# phoneme. El parche lo hace task-aware (phoneme Y speech).

log_step "PASO 0: Verificando/parcheando train_ddp.py para soporte de 'speech'"

TRAIN_DDP="${PROJECT_DIR}/train_ddp.py"

if grep -q 'args.task == "phoneme"' "$TRAIN_DDP" 2>/dev/null; then
  log_ok "train_ddp.py ya está parcheado para speech. Sin cambios."
else
  log_warn "train_ddp.py necesita parche para soporte de 'speech'. Aplicando..."

  # Backup antes de modificar
  cp "$TRAIN_DDP" "${TRAIN_DDP}.bak_$(date +%Y%m%d_%H%M%S)"
  log_ok "Backup guardado: ${TRAIN_DDP}.bak_*"

  # Parche Python: reemplaza el bloque de class_weights hardcodeado a phoneme
  # por una versión task-aware (phoneme + speech)
  python3 - << 'PYEOF'
import sys

TRAIN_DDP = sys.argv[1] if len(sys.argv) > 1 else "train_ddp.py"

with open(TRAIN_DDP, 'r') as f:
    content = f.read()

OLD = """        # Cargar solo las filas de phonemes
        all_phonemes = []
        for tsv in tsv_files:
            try:
                df = pd.read_csv(tsv, sep='\\t')
                # Filtrar solo eventos de tipo 'phoneme' (no 'word', 'silence', etc.)
                phoneme_rows = df[df['type'] == 'phoneme'] if 'type' in df.columns else df
                if 'value' in phoneme_rows.columns:
                    all_phonemes.extend(phoneme_rows['value'].tolist())
            except Exception as e:
                print(f\"  [WARN] Error leyendo {tsv}: {e}\", flush=True)
        
        print(f\"[INFO] Total fonemas en TSVs: {len(all_phonemes)}\", flush=True)
        
        # Mapear strings a índices usando el mapping del propio dataset
        if hasattr(train_pnpl, 'phoneme_to_id'):
            ph_map = train_pnpl.phoneme_to_id
        elif hasattr(train_pnpl, 'label_map'):
            ph_map = train_pnpl.label_map
        else:
            # Fallback: contar strings únicos
            unique = sorted(set(all_phonemes))
            ph_map = {p: i for i, p in enumerate(unique)}
        
        # Contar ocurrencias por clase
        counts = np.zeros(n_classes, dtype=np.float64)
        for ph in all_phonemes:
            if ph in ph_map and ph_map[ph] < n_classes:
                counts[ph_map[ph]] += 1"""

NEW = """        # Contar distribución de clases según la tarea (phoneme vs speech)
        counts = np.zeros(n_classes, dtype=np.float64)

        if args.task == "phoneme":
            # Fonemas: labels son strings (e.g. "AE", "T") → mapear a índice
            all_labels_str = []
            for tsv in tsv_files:
                try:
                    df = pd.read_csv(tsv, sep='\\t')
                    rows = df[df['type'] == 'phoneme'] if 'type' in df.columns else df
                    if 'value' in rows.columns:
                        all_labels_str.extend(rows['value'].tolist())
                except Exception as e:
                    print(f\"  [WARN] Error leyendo {tsv}: {e}\", flush=True)

            print(f\"[INFO] Total fonemas en TSVs: {len(all_labels_str)}\", flush=True)

            if hasattr(train_pnpl, 'phoneme_to_id'):
                label_map = train_pnpl.phoneme_to_id
            elif hasattr(train_pnpl, 'label_map'):
                label_map = train_pnpl.label_map
            else:
                unique = sorted(set(all_labels_str))
                label_map = {p: i for i, p in enumerate(unique)}

            for lbl in all_labels_str:
                if lbl in label_map and label_map[lbl] < n_classes:
                    counts[label_map[lbl]] += 1

        elif args.task == "speech":
            # Speech: binario (0=no-speech, 1=speech). Labels son enteros.
            all_labels_int = []
            for tsv in tsv_files:
                try:
                    df = pd.read_csv(tsv, sep='\\t')
                    rows = df[df['type'] == 'speech'] if 'type' in df.columns else df
                    if 'value' in rows.columns:
                        all_labels_int.extend(rows['value'].astype(int).tolist())
                except Exception as e:
                    print(f\"  [WARN] Error leyendo {tsv}: {e}\", flush=True)

            print(f\"[INFO] Total eventos speech en TSVs: {len(all_labels_int)}\", flush=True)
            for lbl in all_labels_int:
                if 0 <= lbl < n_classes:
                    counts[lbl] += 1
            if counts.sum() > 0:
                print(f\"[INFO] Speech counts: no-speech={counts[0]:.0f}, speech={counts[1]:.0f}\", flush=True)

        else:
            print(f\"[WARN] Tarea desconocida '{args.task}' para pesos de clase. Usando uniformes.\", flush=True)"""

if OLD not in content:
    print("ERROR: Bloque a parchear no encontrado. ¿Ya fue parcheado?")
    sys.exit(1)

content = content.replace(OLD, NEW)

with open(TRAIN_DDP, 'w') as f:
    f.write(content)

print(f"Parche aplicado OK a {TRAIN_DDP}")
PYEOF
  python3 - "$TRAIN_DDP"

  if grep -q 'args.task == "phoneme"' "$TRAIN_DDP"; then
    log_ok "Parche aplicado correctamente."
  else
    log_err "El parche falló. Revisa train_ddp.py manualmente."
    exit 1
  fi
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
    touch "$STATS_SENTINEL"
    continue
  fi

  # Lanzar precompute con la tarea correcta (override del command del compose)
  docker compose run --rm \
    --no-deps \
    -e "TASK_OVERRIDE=${TASK}" \
    precompute_stats \
    /workspace/precompute_stats.py \
      --data_path "${DATA_PATH}" \
      --task "${TASK}" \
    2>&1 | tee "${PROJECT_DIR}/logs/precompute_${TASK}.log"

  EXIT_CODE=${PIPESTATUS[0]}
  if [[ $EXIT_CODE -eq 0 ]]; then
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
    log "  [DRY-RUN] docker compose run --rm meg_training_job train_ddp.py \\"
    log "    --task ${TASK} --backbone ${BACKBONE} --strategy ${STRATEGY} \\"
    log "    --n_epochs ${N_EPOCHS} --batch_size ${BATCH_SIZE} \\"
    log "    --output_dir ${OUTPUT_DIR} --checkpoint_dir ${CKPT_DIR} \\"
    log "    --checkpoint_every ${CHECKPOINT_EVERY} --resume_from ${RESUME_FROM}"
    PASSED+=("$EXP [dry-run]")
    continue
  fi

  START_TS=$(date +%s)

  # ── Lanzar job de entrenamiento ─────────────────────────────────────────────
  docker compose run --rm \
    --no-deps \
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
      --resume_from   "${RESUME_FROM}" \
    2>&1 | tee "$JOB_LOG"

  EXIT_CODE=${PIPESTATUS[0]}
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
# PASO 3: RESUMEN FINAL
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
# PASO 4: TABLA COMPARATIVA DE RESULTADOS
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
print(f"\n{'Experimento':<45} {'Task':<10} {'Backbone':<18} {'Strategy':<12} {'Test F1':>8} {'Bal Acc':>8}")
print("─" * 110)
for r in rows:
    exp = f"{r.get('task','?')}__{r.get('backbone','?')}__{r.get('strategy','?')}"
    print(
        f"  {exp:<43} "
        f"{r.get('task','?'):<10} "
        f"{r.get('backbone','?'):<18} "
        f"{r.get('strategy','?'):<12} "
        f"{r.get('test_f1_macro', 0):.4f}   "
        f"{r.get('test_balanced_acc', 0):.4f}"
    )
print()
PYEOF

log ""
log_ok "Sweep finalizado."
