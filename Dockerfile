FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY requirements.txt .
RUN uv venv --python python3.12 /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install -r requirements.txt

COPY . .

CMD ["python", "brainstorm/train_criss_cross_multi.py", "--config-name=train_criss_cross_multi_50hz_med"]
