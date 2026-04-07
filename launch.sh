#!/bin/bash
# ==============================================================================
# launch.sh — Lanzamiento seguro del job MEG en servidor compartido
# ==============================================================================
#
# Uso:
#   chmod +x launch.sh
#   ./launch.sh                         # Configuración por defecto
#   ./launch.sh --task speech           # Cambiar tarea
#   ./launch.sh --resume                # Resumir desde último checkpoint
#   ./launch.sh --dry-run               # Verificar sin lanzar
#
# Este script:
#   1. Verifica requisitos del host (Docker, GPU, disco, red)
#   2. Configura límites seguros de recursos
#   3. Lanza el contenedor con logging
#   4. Muestra cómo monitorizar y parar limpiamente
# ==============================================================================

set -euo pipefail  # Salir en cualquier error, variables no definidas, pipes fallidos

# ── Colores para output ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[✓]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[⚠]${NC} $*"; }
log_error() { echo -e "${RED}[✗]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}══ $* ${NC}"; }

# ── Configuración por defecto (editar aquí) ────────────────────────────────────
TASK="phoneme"
BACKBONE="resnet18"
STRATEGY="partial_ft"
N_EPOCHS=30
BATCH_SIZE=32          # Por GPU (batch global = 32 × 2 GPUs = 64)
N_FREQS=96
CHECKPOINT_EVERY=1     # Guardar checkpoint cada epoch
RESUME="none"          # "none" | "latest" | "best" | path al .pt
DRY_RUN=false

# Directorios (relativos al proyecto)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/libribrain_data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"

# Nombre único del contenedor (incluye timestamp para evitar conflictos)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONTAINER_NAME="meg_training_${TASK}_${TIMESTAMP}"

# ── Parsear argumentos ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)      TASK="$2";       shift 2 ;;
        --backbone)  BACKBONE="$2";   shift 2 ;;
        --strategy)  STRATEGY="$2";   shift 2 ;;
        --epochs)    N_EPOCHS="$2";   shift 2 ;;
        --resume)    RESUME="latest"; shift   ;;
        --dry-run)   DRY_RUN=true;    shift   ;;
        *) log_warn "Argumento desconocido: $1"; shift ;;
    esac
done

# ==============================================================================
# PASO 1: VERIFICACIONES PREVIAS AL LANZAMIENTO
# ==============================================================================

log_step "VERIFICACIONES DEL SISTEMA"

# ── 1.1: Docker disponible ────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log_error "Docker no encontrado. Instalar Docker Engine >= 23.0"
    exit 1
fi
DOCKER_VERSION=$(docker --version | grep -oP '[\d.]+' | head -1)
log_info "Docker ${DOCKER_VERSION} detectado"

# ── 1.2: nvidia-container-toolkit ────────────────────────────────────────────
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
    log_error "nvidia-container-toolkit no disponible o GPUs no accesibles desde Docker"
    log_warn "  Instalar con: sudo apt install nvidia-container-toolkit"
    log_warn "  Y reiniciar Docker: sudo systemctl restart docker"
    exit 1
fi
log_info "nvidia-container-toolkit OK"

# ── 1.3: GPUs disponibles ────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "${GPU_COUNT}" -lt 2 ]; then
    log_error "Se necesitan 2 GPUs, solo hay ${GPU_COUNT} disponibles"
    exit 1
fi

# Mostrar estado de GPUs
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader | while IFS=',' read -r idx name mem_used mem_total util; do
    log_info "GPU ${idx}: ${name} | Memoria: ${mem_used}/${mem_total} | Uso: ${util}"
done

# Verificar que las GPUs no están saturadas (>80% uso = probable que otro job corra)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1 | tr -d ' %')
if [ "${GPU_UTIL:-0}" -gt 80 ]; then
    log_warn "GPU 0 al ${GPU_UTIL}% de utilización. Puede haber otro job corriendo."
    log_warn "Consultar con: nvidia-smi"
    read -p "¿Continuar de todas formas? (s/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        log_info "Lanzamiento cancelado."
        exit 0
    fi
fi

# ── 1.4: Espacio en disco ─────────────────────────────────────────────────────
DISK_FREE_GB=$(df -BG "${PROJECT_DIR}" | tail -1 | awk '{print $4}' | tr -d 'G')
MIN_DISK_GB=50  # LibriBrain ~20GB + checkpoints + resultados
if [ "${DISK_FREE_GB}" -lt "${MIN_DISK_GB}" ]; then
    log_error "Espacio libre insuficiente: ${DISK_FREE_GB}GB (mínimo: ${MIN_DISK_GB}GB)"
    exit 1
fi
log_info "Espacio en disco: ${DISK_FREE_GB}GB libres"

# ── 1.5: Puerto DDP libre ─────────────────────────────────────────────────────
DDP_PORT=29500
if ss -tlnp | grep -q ":${DDP_PORT}"; then
    log_warn "Puerto ${DDP_PORT} ocupado. Buscando alternativo..."
    DDP_PORT=$((RANDOM + 20000))
    log_info "Usando puerto alternativo: ${DDP_PORT}"
fi
log_info "Puerto DDP: ${DDP_PORT}"

# ── 1.6: Directorios ─────────────────────────────────────────────────────────
mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"
log_info "Directorios creados/verificados"

# ── 1.7: Imagen Docker construida ────────────────────────────────────────────
if ! docker image inspect meg_training:latest &>/dev/null; then
    log_warn "Imagen Docker no encontrada. Construyendo..."
    docker build -t meg_training:latest "${PROJECT_DIR}" || {
        log_error "Fallo al construir imagen Docker"
        exit 1
    }
fi
log_info "Imagen Docker: meg_training:latest"

# ==============================================================================
# PASO 2: PREPARAR COMANDO DE LANZAMIENTO
# ==============================================================================

log_step "CONFIGURACIÓN DEL JOB"
echo "  Tarea:       ${TASK}"
echo "  Backbone:    ${BACKBONE}"
echo "  Estrategia:  ${STRATEGY}"
echo "  Epochs:      ${N_EPOCHS}"
echo "  Batch/GPU:   ${BATCH_SIZE} (global: $((BATCH_SIZE * 2)))"
echo "  Resume:      ${RESUME}"
echo "  Contenedor:  ${CONTAINER_NAME}"
echo ""

# Comando de entrenamiento dentro del contenedor
TRAIN_CMD=(
    "train_ddp.py"
    "--task"           "${TASK}"
    "--backbone"       "${BACKBONE}"
    "--strategy"       "${STRATEGY}"
    "--n_epochs"       "${N_EPOCHS}"
    "--batch_size"     "${BATCH_SIZE}"
    "--n_freqs"        "${N_FREQS}"
    "--checkpoint_dir" "/workspace/checkpoints"
    "--checkpoint_every" "${CHECKPOINT_EVERY}"
    "--resume_from"    "${RESUME}"
    "--data_path"      "/workspace/libribrain_data"
    "--output_dir"     "/workspace/results"
)

# Comando Docker completo
DOCKER_CMD=(
    docker run
    --name "${CONTAINER_NAME}"
    # ── GPUs ──────────────────────────────────────────────────────────────────
    --gpus '"device=0,1"'               # RTX 6000 índices 0 y 1 explícitamente
    # ── Recursos (servidor compartido: ser considerado) ────────────────────────
    --cpus="16"                          # Máximo 16 cores de CPU
    --memory="64g"                       # Máximo 64 GB RAM
    --memory-swap="64g"                  # Sin swap (evitar thrashing)
    # ── Volúmenes ──────────────────────────────────────────────────────────────
    -v "${DATA_DIR}:/workspace/libribrain_data:ro"
    -v "${CHECKPOINT_DIR}:/workspace/checkpoints:rw"
    -v "${RESULTS_DIR}:/workspace/results:rw"
    -v "${LOGS_DIR}:/workspace/logs:rw"
    # ── Red (necesaria para NCCL/DDP) ─────────────────────────────────────────
    --network host
    --ipc host                           # Memoria compartida para DataLoader workers
    # ── Variables de entorno ──────────────────────────────────────────────────
    -e "MASTER_PORT=${DDP_PORT}"
    -e "MASTER_ADDR=localhost"
    -e "NCCL_DEBUG=WARN"
    -e "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
    # ── Logging Docker ─────────────────────────────────────────────────────────
    --log-driver json-file
    --log-opt max-size=500m
    --log-opt max-file=5
    # ── Seguridad ──────────────────────────────────────────────────────────────
    --security-opt no-new-privileges
    --cap-drop ALL
    # ── Parada limpia ──────────────────────────────────────────────────────────
    --stop-timeout 120                   # 120s para guardar checkpoint antes de SIGKILL
    # ── Modo: detached (background) ────────────────────────────────────────────
    -d
    # ── Imagen ─────────────────────────────────────────────────────────────────
    "meg_training:latest"
    # ── Entrypoint (torchrun) + argumentos ─────────────────────────────────────
    "${TRAIN_CMD[@]}"
)

# ==============================================================================
# PASO 3: LANZAMIENTO
# ==============================================================================

if [ "${DRY_RUN}" = true ]; then
    log_step "DRY RUN — Comando que se ejecutaría:"
    echo "${DOCKER_CMD[*]}"
    log_info "Dry run completado. Sin cambios en el sistema."
    exit 0
fi

log_step "LANZANDO JOB"

# Lanzar contenedor en background
eval "${DOCKER_CMD[*]}" > "${LOGS_DIR}/docker_launch_${TIMESTAMP}.log" 2>&1

CONTAINER_ID=$(docker ps -qf "name=${CONTAINER_NAME}")
if [ -z "${CONTAINER_ID}" ]; then
    log_error "El contenedor no arrancó correctamente."
    log_error "Ver logs: cat ${LOGS_DIR}/docker_launch_${TIMESTAMP}.log"
    exit 1
fi

log_info "Contenedor lanzado: ${CONTAINER_NAME} (ID: ${CONTAINER_ID:0:12})"

# Esperar 10s y verificar que sigue corriendo
sleep 10
if ! docker ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
    log_error "El contenedor se paró inmediatamente. Ver logs:"
    docker logs "${CONTAINER_NAME}" --tail 50
    exit 1
fi

log_info "Job corriendo correctamente ✓"

# ==============================================================================
# PASO 4: INSTRUCCIONES POST-LANZAMIENTO
# ==============================================================================

cat << EOF

${GREEN}════════════════════════════════════════════════════════════${NC}
  JOB LANZADO CORRECTAMENTE
${GREEN}════════════════════════════════════════════════════════════${NC}

  Contenedor:  ${CONTAINER_NAME}

  ${BLUE}Monitorización:${NC}
    # Ver logs en tiempo real:
    docker logs -f ${CONTAINER_NAME}

    # Ver solo últimas 50 líneas:
    docker logs --tail 50 ${CONTAINER_NAME}

    # Estado de las GPUs:
    watch -n 5 nvidia-smi

    # TensorBoard (en otra terminal):
    tensorboard --logdir ${RESULTS_DIR}/tensorboard --port 6006

  ${BLUE}Checkpoints:${NC}
    # Ver checkpoints guardados:
    ls -lh ${CHECKPOINT_DIR}/

    # Ver estado del training:
    cat ${CHECKPOINT_DIR}/training_state.json

  ${BLUE}Parar limpiamente (guarda checkpoint antes de terminar):${NC}
    docker stop ${CONTAINER_NAME}
    # → Envía SIGTERM → el script guarda checkpoint → espera 120s → para

  ${BLUE}Guardar snapshot manual (sin parar):${NC}
    docker kill --signal SIGUSR1 ${CONTAINER_NAME}

  ${BLUE}Reanudar tras parada:${NC}
    ./launch.sh --task ${TASK} --resume

  ${BLUE}Ver uso de recursos:${NC}
    docker stats ${CONTAINER_NAME}

${GREEN}════════════════════════════════════════════════════════════${NC}

EOF
