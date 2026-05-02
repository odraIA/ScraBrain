# MEG-XL-style pretraining with `sensor_mask`

Esta integración añade la base para preentrenar con datasets MEG/EEG que no
tienen el mismo número de sensores. Cada muestra conserva sus canales reales,
el `collate` los rellena con ceros hasta `--max_channels`, y `sensor_mask`
marca qué canales son reales (`True`) y cuáles son padding (`False`).

La máscara se aplica:

- antes de la CWT, para que los canales inexistentes no generen señal;
- después del z-score del escalograma, para volver a anular el padding;
- en `MEGImageModelEndToEnd.forward(..., sensor_mask=...)`, antes de mezclar
  sensores con la proyección `conv`/`mean`/`pca`.

La primera versión de `train_megxl_pretrain.py` implementa pretraining
supervisado compatible con checkpoints de `train_ddp.py`. El masked-token
pretraining completo de MEG-XL queda como siguiente fase, porque requiere
integrar tokenizer/objetivo del código de `megxl/` con este pipeline.

## Pretraining mínimo

```bash
python train_megxl_pretrain.py \
  --datasets libribrain \
  --task phoneme \
  --max_channels 306 \
  --batch_size 128 \
  --epochs 5 \
  --output checkpoints/megxl_pretrain.pt
```

Por ahora el único wrapper implementado de verdad es `libribrain`. La clase
`MultiDatasetWrapper` y el `collate` ya están preparados para añadir Armeni,
MEG-MASC, Broderick u otros datasets cuando esté disponible su API local.

## Fine-tuning desde checkpoint

```bash
torchrun --nproc_per_node=2 train_ddp.py \
  --task phoneme \
  --backbone resnet18 \
  --strategy partial_ft \
  --use_sensor_mask true \
  --max_channels 306 \
  --pretrained_ckpt checkpoints/megxl_pretrain.pt \
  --data_path /workspace/libribrain_data \
  --output_dir /workspace/results \
  --checkpoint_dir /workspace/checkpoints
```

Si el checkpoint tiene una cabeza final con otra forma, el default
`--strict_pretrained_load false` omite esas claves incompatibles y lista
`missing keys`, `unexpected keys` y shape mismatches. Para cargar solo el
backbone:

```bash
torchrun --nproc_per_node=2 train_ddp.py \
  --task phoneme \
  --backbone resnet18 \
  --strategy partial_ft \
  --use_sensor_mask true \
  --max_channels 306 \
  --pretrained_ckpt checkpoints/megxl_pretrain.pt \
  --pretrain_backbone_only true \
  --data_path /workspace/libribrain_data \
  --output_dir /workspace/results \
  --checkpoint_dir /workspace/checkpoints
```

## Flujo antiguo

Sin `--use_sensor_mask true`, `train_ddp.py` mantiene el batch legacy
`(batch_x, batch_y)` y el entrenamiento raw+CWT actual:

```bash
torchrun --nproc_per_node=1 train_ddp.py \
  --task phoneme \
  --backbone resnet18 \
  --strategy partial_ft \
  --n_epochs 1 \
  --batch_size 8 \
  --eval_batch_size 8 \
  --data_path ./libribrain_data \
  --output_dir ./results \
  --checkpoint_dir ./checkpoints
```

## Tests

```bash
python -m pytest tests/test_sensor_mask.py
```

Si `pytest` no está instalado en el entorno:

```bash
python tests/test_sensor_mask.py
```
