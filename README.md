# ScraBrain

Código asociado al Trabajo Final de Máster **“Transferencia de modelos MEG de contexto largo a EEG para clasificación de palabras en lectura natural”**.

El proyecto estudia la adaptación y evaluación de modelos de contexto largo usados originalmente con señales MEG para señales EEG. El objetivo experimental es comprobar su comportamiento en tareas de clasificación y recuperación de palabras en paradigmas de lectura natural y escucha, manteniendo una infraestructura reproducible para preprocesado, entrenamiento, evaluación y análisis.

## Estructura del repositorio

- `brainstorm/`: código principal del paquete, modelos, datamodules, evaluación y entrenamiento.
- `configs/`: configuraciones Hydra para entrenamiento, evaluación, sweeps y fine-tuning.
- `scripts/`: scripts de descarga, preprocesado, entrenamiento, sweeps, evaluación y exportación de resultados.
- `datasets_info/`: metadatos auxiliares y notas de organización de datasets.
- `docs/`: documentación técnica adicional de experimentos concretos.
- `memoria/`: fuentes LaTeX y figuras de la memoria.
- `scripts/legacy/`: launchers heredados mantenidos como referencia, fuera de la raíz.
- `Dockerfile` y `docker-compose*.yml`: entorno reproducible y servicios de ejecución.
- `pyproject.toml` y `uv.lock`: declaración y bloqueo de dependencias.

## Requisitos

- Docker.
- Docker Compose.
- GPU NVIDIA compatible con CUDA.
- NVIDIA Container Toolkit para exponer la GPU a los contenedores.

## Instalación / build

Desde la raíz del repositorio:

```bash
docker compose build
```

El entorno usa `uv` y fija dependencias mediante `uv.lock`. Si se modifican `pyproject.toml` o `uv.lock`, reconstruye la imagen.

## Configuración local

Crea un `.env` local no versionado a partir de `.env.example`:

```bash
cp .env.example .env
```

Completa únicamente las variables necesarias para tu entorno local, como rutas de datasets, checkpoints o tokens de servicios externos. No subas `.env` a Git.

## Datos no incluidos

No se incluyen datasets, checkpoints grandes, logs, cachés ni resultados completos por tamaño, licencias y reproducibilidad práctica. Las rutas esperadas por los servicios Docker son:

- `datasets/`
- `checkpoints/`
- `results/`
- `logs/`
- `data/cache/`
- `hf_cache/`
- `embeddings_cache/`
- `wandb/`

El checkpoint MEG-XL de referencia se espera por defecto en:

```text
checkpoints/baseline/meg-xl-med.ckpt
```

## Ejecución básica

Evaluación EEG de lectura:

```bash
docker compose run --rm eval_eeg_reading
```

Evaluación EEG de escucha:

```bash
docker compose run --rm eval_eeg_listening
```

Evaluación conjunta lectura + escucha:

```bash
docker compose run --rm eval_eeg_reading_listening
```

Entrenamiento continuo lectura + escucha:

```bash
bash scripts/run_eeg_reading_listening_training.sh
```

Entrenamiento lectura + escucha inicializado desde checkpoint MEG-XL:

```bash
bash scripts/run_eeg_reading_listening_megxl_joint.sh
```

Sweep EEG multi-dataset:

```bash
bash scripts/run_eeg_multi_training_sweep.sh
```

Fine-tuning de palabras en `ds004408` con comparación de tres condiciones:

```bash
bash scripts/run_ds004408_three_way_finetuning.sh
```

Consulta `README_REPRODUCIBILIDAD.md` y la documentación bajo `docs/` para la relación entre experimentos, configuraciones y salidas. Las guías históricas se conservan en `docs/legacy_guides/`.

## Reproducibilidad

Las dependencias Python están fijadas en `uv.lock` y el entorno de ejecución se define con Docker. Los resultados exactos dependen de datasets, checkpoints externos, cachés generadas y configuración de GPU que no se versionan en este repositorio.

Para validar la configuración Docker:

```bash
docker compose config >/tmp/scrabrain_compose_config_check.txt
```

## Licencia / uso académico

Este repositorio se distribuye para uso académico y reproducibilidad del TFM. Revisa `LICENSE` y `LICENSE_MEGXL` antes de reutilizar código o artefactos derivados de MEG-XL.
