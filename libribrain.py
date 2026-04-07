"""
LibriBrain SEANet-inspired CNN
==============================
Implementación fiel a Özdogan et al. (2025) para las tareas:
  - Speech Detection  (2 clases: silence / speech)
  - Phoneme Classification (39 clases ARPAbet)

Referencia de arquitectura:
  Table 10 del paper LibriBrain (Appendix D.1)

Uso rápido:
  python libribrain_seanet.py --task speech    --data_dir ./data
  python libribrain_seanet.py --task phoneme   --data_dir ./data
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARQUITECTURA
# ─────────────────────────────────────────────────────────────────────────────

class ResNetBlock1D(nn.Module):
    """
    Bloque residual de la Tabla 10 (filas 2a y 2b).
    Conv1D(k=3) → Conv1D(k=1), con skip connection.
    Sin batch norm (igual que el paper).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.elu(self.conv1(x))
        out = self.conv2(out)
        return F.elu(out + residual)


class LibriBrainCNN(nn.Module):
    """
    CNN ligera inspirada en SEANet, adaptada para MEG.

    Parámetros
    ----------
    n_channels : int
        Número de sensores MEG (306 para el sistema MEGIN de LibriBrain).
    n_classes : int
        2 para speech detection, 39 para phoneme classification.
    dropout : float
        Tasa de dropout antes de la capa de salida (0.5 por defecto).
    """

    def __init__(
        self,
        n_channels: int = 306,
        n_classes: int = 39,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ── Capa 1: proyección espacial de sensores ──────────────────────────
        # 306 → 128  (kernel 7, stride 1, padding same)
        self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=7, padding="same")

        # ── Capas 2a/2b: bloque ResNet ────────────────────────────────────────
        self.resnet = ResNetBlock1D(128)

        # ── Capa 4: submuestreo temporal fuerte ───────────────────────────────
        # kernel 50, stride 25 → divide la longitud temporal por ~25
        self.conv_down = nn.Conv1d(128, 128, kernel_size=50, stride=25)

        # ── Capa 6: refinamiento final ────────────────────────────────────────
        self.conv_refine = nn.Conv1d(128, 128, kernel_size=7, padding="same")

        # ── Cabeza MLP ────────────────────────────────────────────────────────
        # El tamaño exacto se calcula dinámicamente en el primer forward pass.
        self.dropout = nn.Dropout(dropout)
        self._fc_built = False
        self.n_classes = n_classes

    def _build_fc(self, flat_dim: int, device: torch.device):
        """Construye las capas lineales una sola vez tras conocer flat_dim."""
        self.fc1 = nn.Linear(flat_dim, 512).to(device)
        self.fc2 = nn.Linear(512, self.n_classes).to(device)
        self._fc_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_channels, time)
        salida : (batch, n_classes) — logits sin softmax
        """
        # Capa 1
        x = F.elu(self.conv1(x))          # (B, 128, T)

        # Bloque ResNet (ya incluye ELU dentro)
        x = self.resnet(x)                 # (B, 128, T)

        # Submuestreo
        x = F.elu(self.conv_down(x))       # (B, 128, T//25)

        # Refinamiento
        x = F.elu(self.conv_refine(x))     # (B, 128, T//25)

        # Flatten
        x = x.flatten(start_dim=1)        # (B, 128 * T//25)

        # Construir FC la primera vez
        if not self._fc_built:
            self._build_fc(x.shape[1], x.device)

        # MLP
        x = F.relu(self.fc1(x))           # (B, 512)
        x = self.dropout(x)
        x = self.fc2(x)                   # (B, n_classes)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET (pnpl o arrays numpy para pruebas sin descargar datos)
# ─────────────────────────────────────────────────────────────────────────────

class LibriBrainDataset(Dataset):
    """
    Envoltorio para pnpl.datasets.LibriBrainSpeech / LibriBrainPhoneme.
    Si pnpl no está instalado o data_dir es None, genera datos sintéticos.

    Parámetros
    ----------
    task : str
        "speech" o "phoneme".
    partition : str
        "train", "validation" o "test".
    data_dir : str | None
        Ruta donde se descargarán los datos. None → datos sintéticos.
    n_synthetic : int
        Número de muestras sintéticas (solo si data_dir es None).
    """

    # Longitudes de ventana en muestras a 250 Hz
    WINDOW_SAMPLES = {"speech": 625, "phoneme": 125}   # 2.5 s y 0.5 s
    N_CLASSES = {"speech": 2, "phoneme": 39}
    N_CHANNELS = 306

    def __init__(
        self,
        task: str = "phoneme",
        partition: str = "train",
        data_dir: str | None = None,
        n_synthetic: int = 1000,
    ):
        self.task = task
        self.window = self.WINDOW_SAMPLES[task]
        self.n_classes = self.N_CLASSES[task]

        if data_dir is not None:
            self._data = self._load_pnpl(task, partition, data_dir)
        else:
            print(f"[LibriBrainDataset] Usando {n_synthetic} muestras SINTÉTICAS "
                  f"(task={task}, partition={partition})")
            self._data = self._synthetic(n_synthetic)

    # ── Carga real con pnpl ───────────────────────────────────────────────────
    def _load_pnpl(self, task: str, partition: str, data_dir: str):
        try:
            from pnpl.datasets import LibriBrainSpeech, LibriBrainPhoneme
            cls = LibriBrainSpeech if task == "speech" else LibriBrainPhoneme
            return cls(path=data_dir, partition=partition, download=True)
        except ImportError:
            raise ImportError(
                "Instala pnpl con: pip install pnpl\n"
                "O pasa data_dir=None para usar datos sintéticos."
            )

    # ── Datos sintéticos para pruebas rápidas ─────────────────────────────────
    def _synthetic(self, n: int):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, self.N_CHANNELS, self.window)).astype(np.float32)
        y = rng.integers(0, self.n_classes, size=n)
        return list(zip(X, y))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        sample = self._data[idx]
        if isinstance(sample, tuple):
            # Datos sintéticos: (array, label)
            meg, label = sample
            meg = torch.from_numpy(meg)
            label = torch.tensor(label, dtype=torch.long)
        else:
            # Objeto pnpl: expone .meg y .label
            meg = torch.tensor(sample.meg, dtype=torch.float32)
            label = torch.tensor(sample.label, dtype=torch.long)
        return meg, label


# ─────────────────────────────────────────────────────────────────────────────
# 3. PÉRDIDA CON PESOS DE CLASE (fonemas)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    dataset: Dataset,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Pesos inversamente proporcionales a sqrt(n_c), normalizados a suma 1.
    Fórmula exacta del paper: w_c ∝ 1/sqrt(n_c), sum(w_c) = 1.
    """
    counts = torch.zeros(n_classes)
    for _, label in dataset:
        counts[label] += 1
    counts = counts.clamp(min=1)
    weights = 1.0 / counts.sqrt()
    weights = weights / weights.sum()
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for meg, labels in loader:
        meg, labels = meg.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(meg)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * meg.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
) -> dict:
    """Devuelve loss, accuracy y F1-macro (sin sklearn)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = torch.zeros(n_classes, device=device)
    fp = torch.zeros(n_classes, device=device)
    fn = torch.zeros(n_classes, device=device)

    for meg, labels in loader:
        meg, labels = meg.to(device), labels.to(device)
        logits = model(meg)
        loss = criterion(logits, labels)
        total_loss += loss.item() * meg.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for c in range(n_classes):
            tp[c] += ((preds == c) & (labels == c)).sum()
            fp[c] += ((preds == c) & (labels != c)).sum()
            fn[c] += ((preds != c) & (labels == c)).sum()

    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1_per_class = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    f1_macro = f1_per_class.mean().item()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1_macro": f1_macro,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRENAMIENTO COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def train(
    task: str = "phoneme",
    data_dir: str | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    dropout: float = 0.5,
    patience: int = 10,
    seed: int = 0,
    save_path: str = "best_model.pt",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  |  tarea: {task}  |  seed: {seed}")

    # ── Datasets y loaders ───────────────────────────────────────────────────
    n_classes = LibriBrainDataset.N_CLASSES[task]
    train_ds = LibriBrainDataset(task, "train",      data_dir, n_synthetic=4000)
    val_ds   = LibriBrainDataset(task, "validation", data_dir, n_synthetic=500)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = LibriBrainCNN(
        n_channels=LibriBrainDataset.N_CHANNELS,
        n_classes=n_classes,
        dropout=dropout,
    ).to(device)

    # Warm-up: un forward para construir las FC con el tamaño correcto
    dummy = torch.zeros(1, LibriBrainDataset.N_CHANNELS,
                        LibriBrainDataset.WINDOW_SAMPLES[task]).to(device)
    _ = model(dummy)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")

    # ── Pérdida ──────────────────────────────────────────────────────────────
    if task == "phoneme":
        weights = compute_class_weights(train_ds, n_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizador (Adam, igual que el paper) ────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8
    )

    # ── Bucle principal ───────────────────────────────────────────────────────
    best_val_loss = math.inf
    no_improve = 0

    print(f"\n{'Epoch':>6}  {'Train loss':>10}  {'Val loss':>10}  "
          f"{'Val acc':>8}  {'Val F1':>8}")
    print("─" * 54)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, n_classes)
        val_loss = val_metrics["loss"]

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{val_metrics['accuracy']:>8.4f}  {val_metrics['f1_macro']:>8.4f}")

        # Early stopping basado en val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "task": task,
                "n_classes": n_classes,
            }, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping en epoch {epoch} (paciencia={patience})")
                break

    print(f"\nMejor val loss: {best_val_loss:.4f}  →  checkpoint: {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. INFERENCIA
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device | None = None) -> nn.Module:
    """Carga un modelo desde un checkpoint guardado con train()."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LibriBrainCNN(n_classes=ckpt["n_classes"]).to(device)
    # Warm-up para construir FC
    task = ckpt["task"]
    dummy = torch.zeros(1, LibriBrainDataset.N_CHANNELS,
                        LibriBrainDataset.WINDOW_SAMPLES[task]).to(device)
    _ = model(dummy)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, meg: np.ndarray, device: torch.device | None = None):
    """
    Predice clases para un array MEG.

    Parámetros
    ----------
    meg : np.ndarray, shape (batch, 306, T) o (306, T)
    """
    if device is None:
        device = next(model.parameters()).device
    if meg.ndim == 2:
        meg = meg[np.newaxis]
    x = torch.tensor(meg, dtype=torch.float32).to(device)
    logits = model(x)
    probs = logits.softmax(dim=-1)
    preds = probs.argmax(dim=-1)
    return preds.cpu().numpy(), probs.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LibriBrain SEANet CNN")
    parser.add_argument("--task",       default="phoneme",
                        choices=["speech", "phoneme"])
    parser.add_argument("--data_dir",   default=None,
                        help="Ruta a datos LibriBrain. None → datos sintéticos.")
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--dropout",    type=float, default=0.5)
    parser.add_argument("--patience",   type=int, default=10)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--save",       default="best_model.pt")
    args = parser.parse_args()

    train(
        task=args.task,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()