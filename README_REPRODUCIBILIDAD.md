# Reproducibilidad de los experimentos

Este documento recoge la relación entre scripts/configuraciones y los experimentos descritos en la memoria del TFM. Los comandos asumen ejecución desde la raíz del repositorio y un `.env` local basado en `.env.example`.

El foco del TFM es la transferencia de modelos MEG de contexto largo a EEG para clasificación de palabras en lectura natural. En las notas preliminares se identificaron como datasets EEG relevantes SparrKULee, OpenNeuro ds004408 y otros conjuntos de lectura/escucha integrados en los pipelines actuales.

## Mapa de experimentos

| Experimento | Objetivo | Datos esperados | Checkpoint inicial | Script/configuración | Salida esperada |
|---|---|---|---|---|---|
| Evaluación EEG en lectura | Evaluar clasificación/recuperación de palabras sobre datasets EEG de lectura natural. | `datasets/zuco2/data/zuco2`, `datasets/eegdash/data/nm000228` o `EEG_DATASETS_DIR`. | `checkpoints/baseline/meg-xl-med.ckpt` o checkpoint promovido si se activa. | Servicio `eval_eeg_reading` en `docker-compose.yml`; config `configs/eval_criss_cross_word_classification_eeg_reading.yaml`. | `logs/hydra/eval_eeg_reading/`, `logs/`, `results/`. |
| Evaluación EEG en escucha | Evaluar clasificación/recuperación de palabras sobre EEG de escucha. | `datasets/OpenNeuroEEG_ds004408`, `datasets/OpenNeuroEEG_ds007808` o `EEG_DATASETS_DIR`. | `checkpoints/baseline/meg-xl-med.ckpt` o checkpoint promovido si se activa. | Servicio `eval_eeg_listening` en `docker-compose.yml`; config `configs/eval_criss_cross_word_classification_eeg_listening.yaml`. | `logs/hydra/eval_eeg_listening/`, `logs/`, `results/`. |
| Evaluación EEG lectura + escucha | Evaluar conjuntamente datasets de lectura y escucha. | Rutas de lectura y escucha bajo `datasets/` o `EEG_DATASETS_DIR`. | `checkpoints/baseline/meg-xl-med.ckpt` o checkpoint promovido si se activa. | Servicio `eval_eeg_reading_listening` en `docker-compose.yml`; config `configs/eval_criss_cross_word_classification_eeg_reading_listening.yaml`. | `logs/hydra/eval_eeg_reading_listening/`, `logs/`, `results/`. |
| Entrenamiento EEG lectura + escucha | Preentrenar el modelo continuo EEG sobre lectura y escucha. | Datasets EEG en `datasets/`, metadatos en `datasets_info/`, caché en `data/cache/`. | Según configuración, por defecto `CRISS_CROSS_CHECKPOINT`. | `scripts/run_eeg_reading_listening_training.sh`; servicio `eeg_train_reading_listening` en `docker-compose.eeg-reading-listening.yml`; config `configs/train_criss_cross_eeg_reading_listening_continuous.yaml`. | `logs/`, `checkpoints/`, `promotions/`, `results/`, `wandb/`. |
| Entrenamiento EEG en lectura | Preentrenar en lectura natural antes de transferencia o comparación. | `datasets/zuco2/data/zuco2`, `datasets/eegdash/data/nm000228`. | Scratch o MEG-XL según overrides. | `scripts/run_eeg_reading_then_listening_sweep.sh`; config `configs/train_criss_cross_eeg_reading_continuous.yaml`. | `results/eeg_reading_listening_sweep/<timestamp>/`, `logs/eeg_reading_listening_training/`, `checkpoints/eeg_reading_listening_training/`. |
| Entrenamiento EEG en escucha | Continuar o entrenar sobre escucha tras etapa de lectura. | `datasets/OpenNeuroEEG_ds004408`, `datasets/OpenNeuroEEG_ds007808`, `datasets/sparrkulee`. | Checkpoint de la etapa de lectura o inicialización configurada. | `scripts/run_eeg_reading_then_listening_sweep.sh`; config `configs/train_criss_cross_eeg_listening_continuous.yaml`. | `results/eeg_reading_listening_sweep/<timestamp>/`, `logs/eeg_reading_listening_training/`, `checkpoints/eeg_reading_listening_training/`. |
| Pipeline full-band lectura -> escucha | Ejecutar la corrección histórica del flujo full-band: primero lectura, después escucha, pasando explícitamente el checkpoint de lectura a escucha. | Reading: EEGDash + ZuCo. Listening: ds004408/ds007808/SparrKULee según configuración. | Scratch o MEG-XL según modo del launcher. | `scripts/run_eeg_full_band_reading_then_listening_sweep.sh`. | `logs/`, `checkpoints/`, `results/` de la ejecución full-band. |
| Experimentos desde cero | Comparar modelos sin transferencia desde MEG-XL. | Datasets EEG bajo `datasets/`. | Sin pesos MEG-XL efectivos; `model.train_from_scratch=true`. | `scripts/run_eeg_multi_training_sweep.sh`; config `configs/train_criss_cross_eeg_multi_feq.yaml`; también `scripts/run_eeg_reading_then_listening_sweep.sh` con inicialización `scratch`. | `results/eeg_multi_training_sweep/<timestamp>/` o `results/eeg_reading_listening_sweep/<timestamp>/`. |
| Experimentos inicializados desde MEG-XL | Medir transferencia desde checkpoint MEG-XL a EEG. | Datasets EEG bajo `datasets/`. | `checkpoints/baseline/meg-xl-med.ckpt`. | `scripts/run_eeg_multi_training_sweep.sh` con modo `pretrained`; `scripts/run_eeg_reading_listening_megxl_joint.sh`; configs `configs/train_criss_cross_eeg_reading_listening_megxl_cached.yaml` y `configs/train_criss_cross_eeg_reading_listening_megxl_joint.yaml`. | `logs/eeg_reading_listening_megxl/`, `checkpoints/`, `results/`. |
| Fine-tuning final ds004408 | Comparar inicialización aleatoria, checkpoint EEG desde cero y checkpoint EEG inicializado desde MEG-XL en clasificación de palabras. | `datasets/OpenNeuroEEG_ds004408` con BrainVision y TextGrid materializados. | `MEGXL_ARCH_CHECKPOINT`, `SCRATCH_EEG_CHECKPOINT`, `PRETRAINED_EEG_CHECKPOINT`. | `scripts/run_ds004408_three_way_finetuning.sh`; config `configs/ds004408_word_finetuning.yaml`; módulo `scripts.evaluate_ds004408_word_classification`. | `results/ds004408_three_way/<run_id>/`, `logs/word_classification_ds004408_eeg/<run_id>/`, `checkpoints/word_classification_ds004408_eeg/<run_id>/`. |
| Fine-tuning final Weissbart/Alice | Ejecutar comparaciones finales análogas en datasets EEG word-aligned adicionales. | Datasets Weissbart o Alice bajo `datasets/`. | Checkpoints de arquitectura, scratch EEG y EEG preentrenado. | `scripts/run_weissbart_three_way_finetuning.sh`, `scripts/run_alice_three_way_finetuning.sh`; configs `configs/eval_criss_cross_word_classification_weissbart_eeg.yaml`, `configs/eval_criss_cross_word_classification_alice_eeg.yaml`. | `results/weissbart_three_way/`, `results/alice_three_way/`, `logs/`, `checkpoints/`. |

## Artefactos no versionados

Los siguientes artefactos se mantienen fuera de Git:

- `datasets/`: datos originales o derivados sujetos a tamaño/licencias.
- `checkpoints/`: pesos MEG-XL, checkpoints EEG y checkpoints de fine-tuning.
- `logs/`: logs de Hydra, salidas de entrenamiento y trazas de ejecución.
- `results/`: métricas, tablas, figuras y manifiestos generados.
- `data/cache/`, `hf_cache/`, `embeddings_cache/`, `wandb/`, `promotions/`: cachés locales, estado de experimentos y artefactos promovidos.

Para reproducir una fila de la tabla, reconstruye el entorno con `docker compose build`, prepara las rutas esperadas en `.env` y ejecuta el servicio o script indicado.

## Documentación heredada

Las guías anteriores de MEG-XL/EEG-XL y los requirements antiguos se conservan fuera de la raíz:

- `docs/legacy_guides/`: README históricos y notas operativas.
- `docs/legacy_requirements/`: `requirements.txt` y `requirements_megxl.txt` previos a la consolidación con `pyproject.toml` y `uv.lock`.
- `scripts/legacy/`: launchers antiguos que siguen siendo útiles como referencia, pero no forman parte de la ruta principal documentada en `README.md`.
