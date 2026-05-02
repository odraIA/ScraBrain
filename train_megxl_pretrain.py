"""
Supervised MEG-XL-style pretraining with padding + sensor_mask.

This first version implements the structural support needed for multi-dataset
MEG/EEG training: variable channel counts are padded to --max_channels and a
sensor_mask is propagated through raw signals, CWT, and the image model. The
full masked-token objective from MEG-XL is left as the next phase because the
paper code in megxl/ is not wired to this LibriBrain pipeline without a larger
tokenizer/model integration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from meg_transfer_learning_libribrain import (  # noqa: E402
    LibriBrainConfig,
    MEGImageModelEndToEnd,
    MEGPreprocessor,
    TrainingConfig,
    build_optimizer_and_scheduler,
    load_libribrain,
)
from meg_gpu_cwt import CWTLayer, apply_cwt_and_normalize  # noqa: E402
from megxl_adapters.collate import megxl_collate  # noqa: E402
from megxl_adapters.datasets import LibriBrainRawWrapper, MultiDatasetWrapper  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MEG-XL-style supervised pretraining with sensor_mask"
    )
    parser.add_argument("--datasets", default="libribrain",
                        help="Comma-separated datasets. Currently implemented: libribrain.")
    parser.add_argument("--task", default="phoneme", choices=["speech", "phoneme"])
    parser.add_argument("--data_path", default="./libribrain_data")
    parser.add_argument("--max_channels", type=int, default=306)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_freqs", type=int, default=96)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "efficientnet_b0", "vit_tiny", "vit_base"])
    parser.add_argument("--strategy", default="partial_ft",
                        choices=["frozen", "partial_ft", "full_ft"])
    parser.add_argument("--sensor_projection", default="conv", choices=["conv", "mean", "pca"])
    parser.add_argument("--pca_max_fit_samples", type=int, default=65536)
    parser.add_argument("--pretrained", type=str2bool, nargs="?", const=True, default=True,
                        help="Use ImageNet weights for the visual backbone.")
    parser.add_argument("--output", default="checkpoints/megxl_pretrain.pt")
    return parser.parse_args()


def _build_dataset(args, preprocessor) -> tuple[MultiDatasetWrapper, int]:
    dataset_names = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    if not dataset_names:
        raise ValueError("--datasets must contain at least one dataset name")

    datasets = []
    dataset_ids = []
    n_classes: int | None = None

    for name in dataset_names:
        if name != "libribrain":
            raise NotImplementedError(
                f"Dataset {name!r} is not implemented yet. Add a wrapper in "
                "megxl_adapters.datasets once its local API is available."
            )

        pnpl_dataset, current_classes, n_channels = load_libribrain(
            LibriBrainConfig(args.data_path, args.task, "train")
        )
        if args.max_channels < n_channels:
            raise ValueError(
                f"--max_channels={args.max_channels} is smaller than LibriBrain "
                f"channels ({n_channels})."
            )

        datasets.append(
            LibriBrainRawWrapper(
                pnpl_dataset,
                preprocessor=preprocessor,
                task=args.task,
                augment=True,
                dataset_id="libribrain",
                subject_id="libribrain_s0",
            )
        )
        dataset_ids.append("libribrain")
        n_classes = current_classes if n_classes is None else n_classes
        if current_classes != n_classes:
            raise ValueError("All supervised pretraining datasets must share label space.")

    return MultiDatasetWrapper(datasets, dataset_ids=dataset_ids), int(n_classes)


def _loss_and_counts(logits: torch.Tensor, labels: torch.Tensor, criterion):
    if logits.ndim == 2 and logits.shape[1] == 1:
        loss = criterion(logits, labels.float().view_as(logits))
        preds = (torch.sigmoid(logits).squeeze(1) >= 0.5).long()
    else:
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    total = int(labels.numel())
    return loss, correct, total


def save_checkpoint(
    output: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict[str, Any],
    args,
    n_classes: int,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": {
            "datasets": args.datasets,
            "task": args.task,
            "n_classes": n_classes,
            "backbone": args.backbone,
            "strategy": args.strategy,
            "sensor_projection": args.sensor_projection,
            "n_freqs": args.n_freqs,
            "use_sensor_mask": True,
            "max_channels": args.max_channels,
        },
        "use_sensor_mask": True,
        "max_channels": args.max_channels,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__,
    }
    torch.save(checkpoint, output)

    meta_path = output.with_suffix(output.suffix + ".json")
    with open(meta_path, "w") as f:
        json.dump({**checkpoint["config"], **metrics, "epoch": epoch}, f, indent=2)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Pretraining datasets={args.datasets} task={args.task} "
        f"max_channels={args.max_channels}"
    )

    preprocessor = MEGPreprocessor(use_instance_norm=True, clip_std=5.0)
    dataset, n_classes = _build_dataset(args, preprocessor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=partial(megxl_collate, max_channels=args.max_channels),
    )

    cwt_layer = CWTLayer(
        sfreq=250.0,
        n_freqs=args.n_freqs,
        f_min=1.0,
        f_max=125.0,
        B=1.5,
        C=1.0,
    ).to(device)
    model = MEGImageModelEndToEnd(
        backbone_name=args.backbone,
        n_classes=n_classes,
        n_meg_channels=args.max_channels,
        n_freqs=args.n_freqs,
        img_size=224,
        pretrained=args.pretrained,
        strategy=args.strategy,
        sensor_projection=args.sensor_projection,
        pca_max_fit_samples=args.pca_max_fit_samples,
    ).to(device)

    config = TrainingConfig(
        backbone=args.backbone,
        pretrained=args.pretrained,
        strategy=args.strategy,
        n_classes=n_classes,
        sensor_projection=args.sensor_projection,
        pca_max_fit_samples=args.pca_max_fit_samples,
        use_sensor_mask=True,
        max_channels=args.max_channels,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=str(Path(args.output).parent),
        experiment_name="megxl_supervised_pretrain",
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, len(loader))

    if args.task == "speech" and n_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    output = Path(args.output)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            raw = batch["meg"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            sensor_mask = batch["sensor_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                scalogram = apply_cwt_and_normalize(cwt_layer, raw, sensor_mask)

            optimizer.zero_grad()
            logits = model(scalogram, sensor_mask=sensor_mask)
            loss, correct, total = _loss_and_counts(logits, labels, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * total
            total_correct += correct
            total_examples += total
            progress.set_postfix(
                loss=f"{total_loss / max(total_examples, 1):.4f}",
                acc=f"{total_correct / max(total_examples, 1):.4f}",
            )

        if scheduler:
            scheduler.step()

        metrics = {
            "train_loss": float(total_loss / max(total_examples, 1)),
            "train_acc": float(total_correct / max(total_examples, 1)),
            "examples": int(total_examples),
        }
        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"loss={metrics['train_loss']:.4f} | acc={metrics['train_acc']:.4f}"
        )
        save_checkpoint(output, model, optimizer, scheduler, epoch, metrics, args, n_classes)
        print(f"[Checkpoint] Saved compatible checkpoint: {output}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
