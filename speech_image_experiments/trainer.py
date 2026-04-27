from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    jaccard_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class TrainerConfig:
    epochs: int = 20
    stage1_epochs: int = 6
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    lr_backbone_stage2: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 5
    use_cosine: bool = True
    fine_tuning_type: str = "two_stage_ft"  # two_stage_ft | partial_ft | full_ft


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("-inf")
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights_binary(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    labels = labels.astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * 2.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    f1_per_class = f1_score(y_true, pred, average=None, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    out["f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    out["f1_macro"] = out["f1"]
    out["f1_class_0"] = float(f1_per_class[0])
    out["f1_class_1"] = float(f1_per_class[1])
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    out["jaccard"] = float(jaccard_score(y_true, pred, average="binary", zero_division=0))
    out["confusion_matrix"] = cm.tolist()
    out["confusion_matrix_00"] = int(cm[0, 0])
    out["confusion_matrix_01"] = int(cm[0, 1])
    out["confusion_matrix_10"] = int(cm[1, 0])
    out["confusion_matrix_11"] = int(cm[1, 1])

    unique = np.unique(y_true)
    if unique.size > 1:
        out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
    else:
        out["auroc"] = float("nan")
    return out


def _run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer=None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    losses = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.item()))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)
        y_prob.extend(probs)
        y_pred.extend(pred.tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)

    metrics = _compute_metrics(y_true_np, y_prob_np, y_pred_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def _build_optimizer_and_scheduler(model, cfg: TrainerConfig, epochs: int, stage2: bool = False):
    lr_backbone = cfg.lr_backbone_stage2 if stage2 else cfg.lr_backbone
    groups = model.get_optimizer_groups(lr_head=cfg.lr_head, lr_backbone=lr_backbone)
    opt = AdamW(groups, weight_decay=cfg.weight_decay)
    sch = None
    if cfg.use_cosine:
        sch = CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-6)
    return opt, sch


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    class_weights: torch.Tensor,
    cfg: TrainerConfig,
    device: torch.device,
    run_dir: Path,
    wandb_run=None,
) -> Dict[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    stopper = EarlyStopping(patience=cfg.patience)

    if cfg.fine_tuning_type == "two_stage_ft":
        model.set_finetune_mode("frozen")
        opt, sch = _build_optimizer_and_scheduler(model, cfg, epochs=max(1, cfg.stage1_epochs), stage2=False)
        total_epochs = cfg.epochs
        for epoch in range(1, total_epochs + 1):
            if epoch == cfg.stage1_epochs + 1:
                model.set_finetune_mode("partial_ft")
                remaining = max(1, total_epochs - cfg.stage1_epochs)
                opt, sch = _build_optimizer_and_scheduler(model, cfg, epochs=remaining, stage2=True)

            train_m = _run_epoch(model, train_loader, criterion, device, optimizer=opt, grad_clip=cfg.grad_clip)
            val_m = _run_epoch(model, val_loader, criterion, device)
            if sch is not None:
                sch.step()

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_m["loss"],
                    "train/f1_macro": train_m["f1_macro"],
                    "train/f1_class_0": train_m["f1_class_0"],
                    "train/f1_class_1": train_m["f1_class_1"],
                    "val/loss": val_m["loss"],
                    "val/f1_macro": val_m["f1_macro"],
                    "val/f1_class_0": val_m["f1_class_0"],
                    "val/f1_class_1": val_m["f1_class_1"],
                    "val/balanced_accuracy": val_m["balanced_accuracy"],
                    "val/auroc": val_m["auroc"],
                    "val/jaccard": val_m["jaccard"],
                })

            if stopper.step(val_m["f1_macro"], model):
                break

    elif cfg.fine_tuning_type in {"partial_ft", "full_ft"}:
        model.set_finetune_mode(cfg.fine_tuning_type)
        opt, sch = _build_optimizer_and_scheduler(model, cfg, epochs=cfg.epochs, stage2=False)
        for epoch in range(1, cfg.epochs + 1):
            train_m = _run_epoch(model, train_loader, criterion, device, optimizer=opt, grad_clip=cfg.grad_clip)
            val_m = _run_epoch(model, val_loader, criterion, device)
            if sch is not None:
                sch.step()

            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_m["loss"],
                    "train/f1_macro": train_m["f1_macro"],
                    "train/f1_class_0": train_m["f1_class_0"],
                    "train/f1_class_1": train_m["f1_class_1"],
                    "val/loss": val_m["loss"],
                    "val/f1_macro": val_m["f1_macro"],
                    "val/f1_class_0": val_m["f1_class_0"],
                    "val/f1_class_1": val_m["f1_class_1"],
                    "val/balanced_accuracy": val_m["balanced_accuracy"],
                    "val/auroc": val_m["auroc"],
                    "val/jaccard": val_m["jaccard"],
                })

            if stopper.step(val_m["f1_macro"], model):
                break
    else:
        raise ValueError(f"fine_tuning_type desconocido: {cfg.fine_tuning_type}")

    stopper.restore(model)

    val_metrics = _run_epoch(model, val_loader, criterion, device)
    test_metrics = _run_epoch(model, test_loader, criterion, device)

    payload = {
        "val_loss": val_metrics["loss"],
        "val_f1": val_metrics["f1"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_class_0": val_metrics["f1_class_0"],
        "val_f1_class_1": val_metrics["f1_class_1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_auroc": val_metrics["auroc"],
        "val_jaccard": val_metrics["jaccard"],
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "val_confusion_matrix_00": val_metrics["confusion_matrix_00"],
        "val_confusion_matrix_01": val_metrics["confusion_matrix_01"],
        "val_confusion_matrix_10": val_metrics["confusion_matrix_10"],
        "val_confusion_matrix_11": val_metrics["confusion_matrix_11"],
        "test_loss": test_metrics["loss"],
        "test_f1": test_metrics["f1"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_class_0": test_metrics["f1_class_0"],
        "test_f1_class_1": test_metrics["f1_class_1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_auroc": test_metrics["auroc"],
        "test_jaccard": test_metrics["jaccard"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_confusion_matrix_00": test_metrics["confusion_matrix_00"],
        "test_confusion_matrix_01": test_metrics["confusion_matrix_01"],
        "test_confusion_matrix_10": test_metrics["confusion_matrix_10"],
        "test_confusion_matrix_11": test_metrics["confusion_matrix_11"],
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
