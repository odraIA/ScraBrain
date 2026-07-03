# Fine-tuning de palabras con OpenNeuro ds004408

El flujo nuevo compara cuatro inicializaciones manteniendo exactamente el mismo preprocesamiento, split y configuración de fine-tuning:

1. arquitectura CrissCross inicializada aleatoriamente;
2. checkpoint del currículo EEG entrenado desde cero;
3. checkpoint inicializado desde MEG-XL que utiliza la fila específica de EEG (`eeg2`);
4. checkpoint inicializado desde MEG-XL que reutiliza para EEG la fila de magnetómetros (`eeg1`).

En las cuatro ejecuciones, los electrodos de ds004408 conservan el tipo físico `eeg`, cuyo identificador es 2. La diferencia entre `eeg1` y `eeg2` afecta únicamente a la fila consultada en `sensor_type_layer`: `eeg1` usa la fila 1 y las demás variantes usan la fila 2.

## Preparación `word_aligned`

Cada `TextGrid` de ds004408 incluye un nivel de palabras y otro de fonemas. El cargador específico selecciona únicamente `word`/`words`, aplica el montaje `biosemi128`, elimina los canales marcados como `bad`, filtra a 0.1–40 Hz, remuestrea a 50 Hz y crea ventanas de 3 s iniciadas 0.5 s antes del onset de cada palabra. Cada muestra concatena 50 ventanas consecutivas.

```bash
docker compose run --rm --no-deps eval_eeg_listening \
  uv run --no-sync python scripts/prepare_ds004408_word_aligned.py \
    --root ./datasets/OpenNeuroEEG_ds004408 \
    --cache-dir ./data/cache/ds004408_word_aligned_v2 \
    --output-dir ./results/ds004408_word_aligned \
    --warm-cache
```

La preparación genera `summary.json`, `alignment_report.json` y `word_aligned_manifest.csv`. El lanzador completo ejecuta esta preparación una sola vez antes de iniciar los fine-tunings.

## Lanzamiento paralelo de las cuatro variantes

Los checkpoints predeterminados corresponden al currículo generado en `20260629_004853` y a las carpetas terminadas en `language_seed42`.

Por defecto, el script detecta las GPU que están libres al comenzar y permite dos fine-tunings simultáneos por GPU. Con dos GPU libres, los cuatro experimentos se lanzan a la vez, distribuidos como `0, 1, 0, 1`.

```bash
NUM_EPOCHS=50 BATCH_SIZE=1 TRAIN_PCT=1.0 \
  bash scripts/run_ds004408_four_way_finetuning.sh
```

Para seleccionar expresamente las dos GPU:

```bash
GPU_LIST=0,1 JOBS_PER_GPU=2 \
NUM_EPOCHS=50 BATCH_SIZE=1 TRAIN_PCT=1.0 \
  bash scripts/run_ds004408_four_way_finetuning.sh
```

Para una ejecución más conservadora, con un único proceso por GPU:

```bash
GPU_LIST=0,1 JOBS_PER_GPU=1 \
  bash scripts/run_ds004408_four_way_finetuning.sh
```

Si solo se detecta una GPU libre y se mantiene `JOBS_PER_GPU=2`, el script ejecuta dos experimentos simultáneos y después los otros dos. La detección automática considera libre una GPU con, como máximo, 1024 MiB utilizados y 10 % de utilización. Estos umbrales se pueden cambiar mediante `FREE_GPU_MAX_MEMORY_MIB` y `FREE_GPU_MAX_UTILIZATION`.

La creación de la caché compartida de embeddings T5 está protegida mediante un bloqueo de fichero, de modo que solo un proceso la genera y los demás esperan para reutilizarla.

Se pueden sustituir los checkpoints sin modificar el script:

```bash
CURRICULUM_ROOT=./checkpoints/eeg_language_curriculum_three_models/20260629_004853 \
FROM_SCRATCH_EEG_CHECKPOINT=/ruta/checkpoint_from_scratch.pt \
MEGXL_EEG2_CHECKPOINT=/ruta/checkpoint_megxl_eeg2.pt \
MEGXL_EEG1_CHECKPOINT=/ruta/checkpoint_megxl_eeg1.pt \
  bash scripts/run_ds004408_four_way_finetuning.sh
```

Seguimiento:

```bash
RUN_ID=$(cat ds004408_four_way.latest)
tail -f "logs/ds004408_four_way_${RUN_ID}.log"
column -ts $'\t' "results/ds004408_four_way/${RUN_ID}/runs.tsv"
cat "results/ds004408_four_way/${RUN_ID}/current_containers.tsv"
watch -n 2 nvidia-smi
```

El fichero `runs.tsv` registra el identificador del embedding y la GPU empleada por cada ejecución. El informe combinado se genera después de que terminen los cuatro procesos y se guarda en:

```text
results/ds004408_four_way/<RUN_ID>/ds004408_four_way_test_metrics.csv
```

El script anterior de tres vías se conserva para reproducir los experimentos previos.
