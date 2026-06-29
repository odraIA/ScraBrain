# Fine-tuning de palabras con OpenNeuro ds004408

El flujo reproduce la comparación de tres inicializaciones utilizada con Weissbart y Alice EEG:

1. arquitectura CrissCross inicializada aleatoriamente;
2. checkpoint EEG entrenado desde cero con el currículo reading → listening;
3. checkpoint EEG inicializado desde MEG-XL y entrenado con el mismo currículo.

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

La preparación genera `summary.json`, `alignment_report.json` y `word_aligned_manifest.csv`.

## Lanzamiento completo

```bash
bash scripts/run_ds004408_three_way_finetuning.sh
```

La preparación y los tres entrenamientos se ejecutan secuencialmente en una GPU. El split 80/10/10 se hace mediante hash de la secuencia completa de 50 palabras, de modo que el mismo fragmento oído por distintos sujetos no aparece en particiones diferentes.

```bash
GPU=1 NUM_EPOCHS=50 BATCH_SIZE=1 TRAIN_PCT=1.0 \
  bash scripts/run_ds004408_three_way_finetuning.sh
```

Seguimiento:

```bash
RUN_ID=$(cat ds004408_three_way.latest)
tail -f "logs/ds004408_three_way_${RUN_ID}.log"
column -ts $'\t' "results/ds004408_three_way/${RUN_ID}/runs.tsv"
```
