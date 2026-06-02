FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache/transformers \
    MPLCONFIGDIR=/workspace/hf_cache/matplotlib \
    WANDB_DIR=/workspace/wandb \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv pip install --no-deps -e .

CMD ["uv", "run", "--no-sync", "python", "-m", "brainstorm.evaluate_criss_cross_word_classification", "--config-name=eval_criss_cross_word_classification_libribrain"]
