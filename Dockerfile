# ==============================================================================
# Dockerfile — MEG Transfer Learning con PyTorch + NVIDIA RTX 6000
# ==============================================================================
#
# Base: imagen oficial PyTorch con CUDA 12.1 y cuDNN 8 (compatible RTX 6000 Ada)
# Para Quadro RTX 6000 (Turing, CUDA 11.x) cambiar a:
#   pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
#
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# ── Metadatos ──────────────────────────────────────────────────────────────────
LABEL maintainer="rdiaper"
LABEL description="MEG Transfer Learning — LibriBrain"
LABEL version="1.0"

# ── Variables de entorno ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Optimizaciones CUDA
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    # Para RTX 6000 Ada: sm_89. Para Quadro RTX 6000 (Turing): "7.5"
    # Desactivar tokenizers paralelos (evita deadlocks con DataLoader)
    TOKENIZERS_PARALLELISM=false \
    # Directorio de trabajo dentro del contenedor
    WORKDIR_PATH=/workspace

# ── Instalar dependencias del sistema ─────────────────────────────────────────
# Mínimas: sin GUI, sin librerías X11 innecesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        htop \
        tmux \
        vim \
        libhdf5-dev \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Crear usuario no-root (SEGURIDAD: nunca ejecutar como root en producción) ──
# UID/GID 1000 para coincidir con usuario típico del host (evita problemas de permisos)

# ── Directorio de trabajo ──────────────────────────────────────────────────────
WORKDIR ${WORKDIR_PATH}

# ── Copiar requirements primero (cachear capa si no cambian) ───────────────────
COPY requirements.txt .

# ── Instalar dependencias Python ───────────────────────────────────────────────
# Se instalan como root, luego se cede a meguser
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copiar código del proyecto ─────────────────────────────────────────────────
# Se copia al final para no invalidar la caché de pip en cada cambio de código
COPY . .

RUN useradd -m -u 1000 meguser
# ── Permisos correctos ─────────────────────────────────────────────────────────
RUN chown -R meguser:meguser ${WORKDIR_PATH}

# ── Crear directorios de trabajo con permisos adecuados ───────────────────────
RUN mkdir -p \
    ${WORKDIR_PATH}/checkpoints \
    ${WORKDIR_PATH}/results \
    ${WORKDIR_PATH}/logs \
    ${WORKDIR_PATH}/libribrain_data \
    && chown -R meguser:meguser ${WORKDIR_PATH}

# ── Cambiar a usuario no-root ─────────────────────────────────────────────────
USER meguser

# ── Health check: verificar que GPU es accesible ──────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

# ── Punto de entrada por defecto ───────────────────────────────────────────────
# torchrun gestiona el lanzamiento DDP automáticamente
ENTRYPOINT ["torchrun", "--nproc_per_node=2", "--nnodes=1"]
CMD ["train_ddp.py", "--help"]
