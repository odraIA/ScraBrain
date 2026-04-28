# Speech Image Experiments (MEG -> TF image -> ImageNet backbone)

Este flujo implementa exclusivamente:

- `MEG -> CWT/Morlet -> tensor [sensores, frecuencias, tiempo]`
- proyección a 3 canales (`learnable_1x1_projection`, `pca3_projection`, `current_image_projection`)
- backbone visual preentrenado en ImageNet (`resnet18` o `vit_tiny`)
- clasificación `speech` binaria

No hay modelos sobre señal cruda en este runner.

## Lanzamiento rápido

- Baseline ResNet18: `bash scripts/experiments/baseline_image_resnet18.sh`
- Baseline ViT-Tiny: `bash scripts/experiments/baseline_image_vittiny.sh`
- Ablación proyección: `bash scripts/experiments/ablation_projection.sh`
- Ablación fine-tuning: `bash scripts/experiments/ablation_finetuning.sh`
- Ablación longitud ventana: `bash scripts/experiments/ablation_window_length.sh`
- Ablación augmentations: `bash scripts/experiments/ablation_augmentations.sh`
- Variante low-frequency bias: `bash scripts/experiments/ablation_low_freq_bias.sh`

## Integración con sweep/monitor existentes

- Sweep completo A–F con el flujo estándar: `bash run_sweep.sh --speech-image`
- Sweep A–F desacoplado de la terminal: `bash run_sweep.sh --speech-image --detach`
- Relanzar experimentos desde cero aunque exista `.exp_done_*`: `bash run_sweep.sh --speech-image --rerun`
- Relanzar desde cero y dejarlo en background: `bash run_sweep.sh --speech-image --rerun --detach`
- Recalcular también stats aunque exista `.precompute_done_*`: `bash run_sweep.sh --speech-image --rerun-precompute`
- Añadir variante low-freq en el sweep: `bash run_sweep.sh --speech-image --low-freq-bias`
- Monitor en paralelo (igual que antes): `docker compose up -d monitor`

`run_sweep.sh` escribe `.sweep_mode=speech_image`, y `monitor_server.py` cambia automáticamente a descubrimiento dinámico de experimentos para mostrar este modo.
Con `--detach`, el coordinador queda bajo `nohup`, deja PID en `.sweep_coordinator_speech_image.pid` y symlinks a los logs más recientes en `logs/latest_speech_image_*`.

`--resume` reanuda desde checkpoints existentes; `--rerun` ignora los sentinels `.exp_done_*`, elimina el sentinel del experimento antes de lanzarlo y entrena desde cero. Los sentinels de precompute se respetan salvo que añadas `--rerun-precompute`.

## CLI directa

```bash
python run_speech_image_experiments.py \
  --experiment baseline_image_resnet18 \
  --data_path ./libribrain_data \
  --output_dir ./results/speech_image_experiments \
  --epochs 20 \
  --stage1_epochs 6 \
  --batch_size 32 \
  --num_workers 4 \
  --seeds 42,43,44 \
  --tf_variant full_band_tf
```

`--tf_variant` acepta:

- `full_band_tf`
- `low_freq_biased_tf`

## Salidas

Para cada experimento se guardan:

- `*_all_runs.csv` y `*_all_runs.json`
- `*_comparison_table.csv` y `*_comparison_table.json`
- `run_summary.json` + `metrics.json` por seed

Métricas reportadas en validación y test:

- `F1`
- `balanced_accuracy`
- `AUROC`
- `Jaccard`
- `loss`

## wandb

Si está instalado, activar con `--use_wandb`.
