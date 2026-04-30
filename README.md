# ScraBrain

Final project of AI Master about decoding EEG and MEG for imagined speech
recognition.

## Entorno Docker con uv

La imagen base es `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, por lo que
PyTorch, CUDA y `torchrun` vienen de la imagen. Las dependencias del proyecto se
declaran en `pyproject.toml` y se fijan en `uv.lock`.

Durante el build se crea la venv en `/workspace/.venv` con acceso a los
`site-packages` de la imagen base. Esto permite que:

```bash
python
torchrun
timm
pnpl
sklearn
h5py
pywt
```

se resuelvan correctamente sin reinstalar `torch`, `torchvision` ni `torchaudio`
desde PyPI.

## Token de Hugging Face

`docker-compose.yml` lee el token desde `HF_TOKEN=${HF_TOKEN}`. Define el token
en un `.env` local no versionado:

```bash
HF_TOKEN=hf_xxx
```

`.env` ya esta incluido en `.gitignore`.

## Build

```bash
docker compose build
```

Si cambias `pyproject.toml` o `uv.lock`, reconstruye la imagen. Si ya existia el
volumen de la venv y quieres forzar que se repueble desde la imagen nueva:

```bash
docker compose down -v
docker compose build
```

## Sweep principal

El flujo habitual sigue siendo:

```bash
bash run_sweep.sh --detach
```

Para revisar lo que lanzaria sin ejecutar contenedores:

```bash
bash run_sweep.sh --dry-run
```

`run_sweep.sh` mantiene el uso de `docker compose run` detached sobre los
servicios `precompute_stats` y `meg_training_job`.

## Logs del sweep

Ver el coordinador:

```bash
tail -f logs/latest_classic_coordinator.log
```

Ver el log global del sweep:

```bash
tail -f logs/latest_classic_sweep.log
```

Ver un experimento concreto:

```bash
tail -f logs/speech__resnet18__partial_ft.log
```

## Parar el coordinador

El modo `--detach` escribe el PID en `.sweep_coordinator_classic.pid`:

```bash
kill "$(cat .sweep_coordinator_classic.pid)"
```

Para el sweep speech-image, usa `.sweep_coordinator_speech_image.pid`.

## Precompute y entrenamiento manual

Precalcular stats:

```bash
docker compose run --rm precompute_stats
```

Comprobar el entrypoint de entrenamiento:

```bash
docker compose run --rm meg_training_job train_ddp.py --help
```

## Shell de depuracion

Levantar una shell persistente:

```bash
docker compose up -d dev_shell
docker compose exec dev_shell bash
```

Comprobar CUDA y paquetes dentro del contenedor:

```bash
python - <<'PY'
import torch
import timm
import pnpl
import sklearn
import h5py
import pywt

print("python ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
PY
```
