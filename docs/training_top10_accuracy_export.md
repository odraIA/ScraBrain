# Exportar top-10 accuracy durante entrenamiento

El exportador está en `scripts/export_training_top10_accuracy.py`. En el servidor,
la forma recomendada es lanzarlo dentro del contenedor:

```bash
bash scripts/export_training_top10_accuracy_docker.sh
```

Por defecto busca historiales en:

- `logs/eeg_language_curriculum_three_models`
- `logs/word_classification_ds004408_four_way`
- `logs/word_classification_ds004408_eeg`
- `logs/word_classification_ds004408_three_way`

La salida se escribe en `results/training_top10_accuracy/<RUN_ID>/` e incluye:

- `top10_training_curves_long.csv`
- `top10_training_curves_manifest.csv`
- `top10_training_curves_manifest.json`
- CSVs pivotados en `comparison/` y `per_run/`
- figuras `.png` y `.pdf` cuando haya métricas compatibles

## Ejemplos

Exportar solo una ejecución de ds004408 four-way:

```bash
TOP10_EXPORT_OUTPUT_DIR=results/training_top10_accuracy/ds004408_20260703 \
  bash scripts/export_training_top10_accuracy_docker.sh \
    --scan-root logs/word_classification_ds004408_four_way \
    --include-pattern 20260703
```

Exportar un run concreto:

```bash
bash scripts/export_training_top10_accuracy_docker.sh \
  --run-dir logs/word_classification_ds004408_four_way/<RUN_ID>/megxl_eeg2
```

Exportar currículo y ds004408 a una carpeta concreta:

```bash
TOP10_EXPORT_OUTPUT_DIR=results/training_top10_accuracy/curriculum_ds004408 \
  bash scripts/export_training_top10_accuracy_docker.sh \
    --scan-root logs/eeg_language_curriculum_three_models \
    --scan-root logs/word_classification_ds004408_four_way
```

Si solo quieres CSVs, sin PNG/PDF:

```bash
bash scripts/export_training_top10_accuracy_docker.sh --no-figures
```

## Notas

El script exporta lo que esté guardado. En ds004408, si
`evaluation.evaluate_test_during_training=false`, durante entrenamiento habrá
validación por época, pero no test por época. El manifiesto marca cada historial
sin columnas top-10 solicitadas como `no_requested_top10_columns`.
