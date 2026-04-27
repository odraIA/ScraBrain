#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from meg_transfer_learning_libribrain import LibriBrainConfig, MEGPreprocessor, load_libribrain
from speech_image_experiments import (
    TFImageConfig,
    TFImageGenerator,
    AugmentationConfig,
    MEGTFDataset,
    fit_pca3_components,
    extract_binary_labels_fast,
    SpeechImageModel,
    TrainerConfig,
    train_and_evaluate,
    compute_class_weights_binary,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Batería de experimentos speech detection con imágenes TF")
    p.add_argument("--experiment", type=str, required=True, choices=[
        "baseline_image_resnet18",
        "baseline_image_vittiny",
        "ablation_projection",
        "ablation_finetuning",
        "ablation_window_length",
        "ablation_augmentations",
    ])
    p.add_argument("--data_path", type=str, default="./libribrain_data")
    p.add_argument("--output_dir", type=str, default="./results/speech_image_experiments")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--stage1_epochs", type=int, default=6)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--tf_variant", type=str, default="full_band_tf", choices=["full_band_tf", "low_freq_biased_tf"])
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="scrabrain-speech-image")
    p.add_argument("--write_final_results", action="store_true", default=True)
    return p.parse_args()


def get_experiment_grid(name: str) -> List[Dict]:
    if name == "baseline_image_resnet18":
        return [{
            "exp_name": "baseline_image_resnet18",
            "backbone": "resnet18",
            "projection_type": "learnable_1x1_projection",
            "fine_tuning_type": "two_stage_ft",
            "window_seconds": 2.0,
            "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
        }]

    if name == "baseline_image_vittiny":
        return [{
            "exp_name": "baseline_image_vittiny",
            "backbone": "vit_tiny",
            "projection_type": "learnable_1x1_projection",
            "fine_tuning_type": "two_stage_ft",
            "window_seconds": 2.0,
            "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
        }]

    if name == "ablation_projection":
        return [
            {
                "exp_name": "ablation_projection_learnable_1x1_projection",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
            {
                "exp_name": "ablation_projection_pca3_projection",
                "backbone": "resnet18",
                "projection_type": "pca3_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
            {
                "exp_name": "ablation_projection_current_image_projection",
                "backbone": "resnet18",
                "projection_type": "current_image_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
        ]

    if name == "ablation_finetuning":
        return [
            {
                "exp_name": "ablation_finetuning_two_stage_ft",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
            {
                "exp_name": "ablation_finetuning_partial_ft",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "partial_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
            {
                "exp_name": "ablation_finetuning_full_ft",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "full_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
        ]

    if name == "ablation_window_length":
        out = []
        for w in [0.5, 1.0, 2.0, 3.0]:
            out.append({
                "exp_name": f"ablation_window_length_{w:.1f}s",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": w,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            })
        return out

    if name == "ablation_augmentations":
        return [
            {
                "exp_name": "ablation_augmentations_none",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "none",
            },
            {
                "exp_name": "ablation_augmentations_temporal_shift",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift",
            },
            {
                "exp_name": "ablation_augmentations_all",
                "backbone": "resnet18",
                "projection_type": "learnable_1x1_projection",
                "fine_tuning_type": "two_stage_ft",
                "window_seconds": 2.0,
                "augmentations": "temporal_shift+frequency_masking+amplitude_jitter",
            },
        ]

    raise ValueError(f"Experimento no soportado: {name}")


def build_augmentation_config(tag: str) -> AugmentationConfig:
    if tag == "none":
        return AugmentationConfig(False, False, False)
    if tag == "temporal_shift":
        return AugmentationConfig(True, False, False)
    return AugmentationConfig(True, True, True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("[INFO] Cargando splits de LibriBrain (session-aware por pnpl)")
    train_pnpl, _, n_channels = load_libribrain(LibriBrainConfig(args.data_path, "speech", "train"))
    val_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, "speech", "validation"))
    test_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, "speech", "test"))

    preprocessor = MEGPreprocessor(use_instance_norm=True, baseline_samples=None, clip_std=5.0)
    tf_cfg = TFImageConfig(tf_variant=args.tf_variant)
    generator = TFImageGenerator(tf_cfg)

    grid = get_experiment_grid(args.experiment)
    rows: List[Dict] = []

    for exp in grid:
        for seed in seeds:
            set_seed(seed)
            run_name = f"{exp['exp_name']}__seed{seed}"
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            aug_cfg = build_augmentation_config(exp["augmentations"])

            pca_components = None
            if exp["projection_type"] == "pca3_projection":
                print(f"[INFO] Ajustando PCA-3 en train para {run_name}")
                pca_components = fit_pca3_components(
                    pnpl_train_dataset=train_pnpl,
                    preprocessor=preprocessor,
                    generator=generator,
                    window_seconds=exp["window_seconds"],
                    max_samples=32,
                    seed=seed,
                )

            train_ds = MEGTFDataset(
                pnpl_dataset=train_pnpl,
                preprocessor=preprocessor,
                generator=generator,
                projection_type=exp["projection_type"],
                split="train",
                window_seconds=exp["window_seconds"],
                augment_cfg=aug_cfg,
                pca_components=pca_components,
                seed=seed,
            )
            val_ds = MEGTFDataset(
                pnpl_dataset=val_pnpl,
                preprocessor=preprocessor,
                generator=generator,
                projection_type=exp["projection_type"],
                split="validation",
                window_seconds=exp["window_seconds"],
                augment_cfg=AugmentationConfig(False, False, False),
                pca_components=pca_components,
                seed=seed,
            )
            test_ds = MEGTFDataset(
                pnpl_dataset=test_pnpl,
                preprocessor=preprocessor,
                generator=generator,
                projection_type=exp["projection_type"],
                split="test",
                window_seconds=exp["window_seconds"],
                augment_cfg=AugmentationConfig(False, False, False),
                pca_components=pca_components,
                seed=seed,
            )

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            labels_train = extract_binary_labels_fast(train_pnpl)
            class_weights = compute_class_weights_binary(labels_train, device=device)

            model = SpeechImageModel(
                backbone=exp["backbone"],
                projection_type=exp["projection_type"],
                n_meg_channels=n_channels,
                img_size=tf_cfg.img_size,
                pretrained=True,
            ).to(device)

            trainer_cfg = TrainerConfig(
                epochs=args.epochs,
                stage1_epochs=min(args.stage1_epochs, args.epochs - 1) if args.epochs > 1 else 1,
                fine_tuning_type=exp["fine_tuning_type"],
                use_cosine=True,
                grad_clip=1.0,
                patience=5,
            )

            wb = None
            if args.use_wandb and WANDB_AVAILABLE:
                wb = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config={**exp, "seed": seed, "tf_variant": args.tf_variant},
                    reinit=True,
                )

            metrics = train_and_evaluate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                cfg=trainer_cfg,
                device=device,
                run_dir=run_dir,
                wandb_run=wb,
            )

            if wb is not None:
                wb.log({f"final/{k}": v for k, v in metrics.items()})
                wb.finish()

            row = {
                "experiment_name": exp["exp_name"],
                "seed": seed,
                "backbone": exp["backbone"],
                "projection_type": exp["projection_type"],
                "fine_tuning_type": exp["fine_tuning_type"],
                "window_seconds": exp["window_seconds"],
                "augmentations": exp["augmentations"],
                "tf_variant": args.tf_variant,
                **metrics,
            }
            rows.append(row)

            with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)

    raw_csv = output_dir / f"{args.experiment}_all_runs.csv"
    raw_json = output_dir / f"{args.experiment}_all_runs.json"
    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    headers = list(rows[0].keys()) if rows else []
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    group_cols = [
        "experiment_name",
        "backbone",
        "projection_type",
        "fine_tuning_type",
        "window_seconds",
        "augmentations",
        "tf_variant",
    ]
    metric_cols = [
        "val_loss",
        "val_f1",
        "val_f1_macro",
        "val_f1_class_0",
        "val_f1_class_1",
        "val_balanced_accuracy",
        "val_auroc",
        "val_jaccard",
        "val_confusion_matrix_00",
        "val_confusion_matrix_01",
        "val_confusion_matrix_10",
        "val_confusion_matrix_11",
        "test_loss",
        "test_f1",
        "test_f1_macro",
        "test_f1_class_0",
        "test_f1_class_1",
        "test_balanced_accuracy",
        "test_auroc",
        "test_jaccard",
        "test_confusion_matrix_00",
        "test_confusion_matrix_01",
        "test_confusion_matrix_10",
        "test_confusion_matrix_11",
    ]

    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[c] for c in group_cols)
        grouped[key].append(row)

    summary_rows: List[Dict] = []
    for key, items in grouped.items():
        s = {k: v for k, v in zip(group_cols, key)}
        for m in metric_cols:
            vals = [float(it[m]) for it in items if not np.isnan(float(it[m]))]
            s[m] = float(np.mean(vals)) if vals else float("nan")
        summary_rows.append(s)

    summary_csv = output_dir / f"{args.experiment}_comparison_table.csv"
    summary_json = output_dir / f"{args.experiment}_comparison_table.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=group_cols + metric_cols)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("\n[OK] Experimentos completados")
    print(f"[OK] Runs: {raw_csv}")
    print(f"[OK] Tabla comparativa: {summary_csv}")
    print("\nTabla final:")
    for row in summary_rows:
        print(row)

    if args.write_final_results and summary_rows:
        # Resultado compacto compatible con run_sweep.sh / monitor_server.py
        best = max(summary_rows, key=lambda r: float(r.get("test_f1_macro", float("-inf"))))
        best_key = tuple(best[c] for c in group_cols)
        matching_runs = [r for r in rows if tuple(r[c] for c in group_cols) == best_key]
        best_run = max(matching_runs, key=lambda r: float(r.get("test_f1_macro", float("-inf"))))
        final_payload = {
            "task": "speech",
            "experiment_name": args.experiment,
            "backbone": best.get("backbone"),
            "projection_type": best.get("projection_type"),
            "fine_tuning_type": best.get("fine_tuning_type"),
            "window_seconds": best.get("window_seconds"),
            "augmentations": best.get("augmentations"),
            "tf_variant": best.get("tf_variant"),
            "test_f1_macro": best.get("test_f1_macro"),
            "test_f1_class_0": best.get("test_f1_class_0"),
            "test_f1_class_1": best.get("test_f1_class_1"),
            "test_balanced_acc": best.get("test_balanced_accuracy"),
            "test_auroc": best.get("test_auroc"),
            "test_jaccard": best.get("test_jaccard"),
            "test_confusion_matrix": best_run.get("test_confusion_matrix"),
            "test_loss": best.get("test_loss"),
            "val_f1_macro": best.get("val_f1_macro"),
            "val_f1_class_0": best.get("val_f1_class_0"),
            "val_f1_class_1": best.get("val_f1_class_1"),
            "val_balanced_acc": best.get("val_balanced_accuracy"),
            "val_auroc": best.get("val_auroc"),
            "val_jaccard": best.get("val_jaccard"),
            "val_confusion_matrix": best_run.get("val_confusion_matrix"),
            "val_loss": best.get("val_loss"),
            "num_runs": len(rows),
        }
        with (output_dir / "final_results.json").open("w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)


if __name__ == "__main__":
    main()
