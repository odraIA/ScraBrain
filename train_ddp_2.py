"""
================================================================================
  train_ddp.py — Entrenamiento Multi-GPU con DDP + Checkpointing Automático
================================================================================

Este script envuelve meg_transfer_learning_libribrain.py con:

  1. DistributedDataParallel (DDP): entrena en paralelo en 2 GPUs RTX 6000
     usando NCCL como backend de comunicación. Cada GPU procesa la mitad
     del batch y los gradientes se sincronizan automáticamente.

  2. Checkpointing automático:
     - Cada N epochs (--checkpoint_every)
     - Al recibir SIGTERM (cuando otro usuario libera las GPUs o el admin
       para el job) → guarda checkpoint antes de morir
     - Al detectar mejora en validación (best model)

  3. Resume desde checkpoint: --resume_from latest | <path>

  4. Logging dual: stdout (visible en docker logs) + TensorBoard

LANZAMIENTO:
  # Dentro del contenedor (gestionado por docker-compose/Dockerfile):
  torchrun --nproc_per_node=2 --nnodes=1 train_ddp.py [args]

  # Directamente en el host (sin Docker, para pruebas):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_ddp.py [args]
================================================================================
"""

import os
import sys
import signal
import argparse
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ── Importar módulos del proyecto ─────────────────────────────────────────────
# Aseguramos que el directorio raíz del proyecto está en el path
sys.path.insert(0, str(Path(__file__).parent))

from meg_transfer_learning_libribrain import (
    LibriBrainConfig,
    MEGPreprocessor,
    MEGToImage,
    MEGImageDataset,
    MEGImageModelEndToEnd,
    TrainingConfig,
    build_optimizer_and_scheduler,
    compute_class_weights_isns,
    load_libribrain,
    EarlyStopping,
    DEVICE,
)

from meg_gpu_cwt import CWTLayer, MEGRawDataset, build_raw_dataloaders, zscore_scalogram


# ==============================================================================
# FUNCIONES DE ENTRENAMIENTO CON CWT EN GPU
# ==============================================================================
# Equivalentes a train_one_epoch / evaluate del fichero original, pero con el
# paso CWT insertado entre el DataLoader y el modelo. El DataLoader entrega
# señales raw (B, 306, T); cwt_layer las convierte a escalogramas en GPU antes
# del forward del modelo.
#
# También aplican normalización ImageNet tras el min-max, necesaria porque
# ResNet18 fue preentrenado con esas estadísticas. MEGImageModelEndToEnd no la
# aplica internamente, así que la hacemos aquí.

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _apply_cwt_and_normalize(cwt_layer, batch_x: torch.Tensor) -> torch.Tensor:
    """
    Aplica CWT + z-score en GPU. Sin gradientes (el CWT no es diferenciable
    en nuestro pipeline; los gradientes fluyen desde el SensorMixer en adelante).

    Args:
        cwt_layer : CWTLayer en el mismo device que batch_x
        batch_x   : (B, 306, T) señales MEG preprocesadas

    Returns:
        scalogram : (B, 306, n_freqs, T) float32, z-score normalizado
    """
    with torch.no_grad():
        scalogram = cwt_layer(batch_x)           # (B, 306, n_freqs, T)
        scalogram = zscore_scalogram(scalogram)  # normalización por banda
    return scalogram


def train_one_epoch_raw(
    model,
    cwt_layer,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    grad_clip: float = 1.0,
):
    """
    Versión de train_one_epoch que acepta señales MEG raw (B, 306, T) y aplica
    CWT en GPU antes del forward del modelo.
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score
    import torch.distributed as dist
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    is_main = (not dist.is_initialized()) or dist.get_rank() == 0

    for batch_idx, (batch_x, batch_y) in enumerate(
        tqdm(loader, desc="Train", leave=False, disable=not is_main)
    ):
        batch_x = batch_x.to(device, non_blocking=True)  # (B, 306, T)
        batch_y = batch_y.to(device, non_blocking=True)

        if batch_idx % 50 == 0 and is_main:
            print(f"  [batch {batch_idx}/{len(loader)}]", flush=True)

        # ── CWT en GPU (sin gradientes) ────────────────────────────────────────
        scalogram = _apply_cwt_and_normalize(cwt_layer, batch_x)  # (B, 306, F, T)

        # ── Forward ───────────────────────────────────────────────────────────
        optimizer.zero_grad()
        logits = model(scalogram)
        loss   = criterion(logits, batch_y)

        # ── Backward ──────────────────────────────────────────────────────────
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return {"loss": avg_loss, "f1_macro": f1, "balanced_acc": bal_acc}


@torch.no_grad()
def evaluate_raw(
    model,
    cwt_layer,
    loader,
    criterion,
    device: torch.device,
):
    """
    Versión de evaluate que acepta señales MEG raw (B, 306, T).
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch_x, batch_y in tqdm(loader, desc="Eval", leave=False):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        scalogram = _apply_cwt_and_normalize(cwt_layer, batch_x)
        logits    = model(scalogram)
        loss      = criterion(logits, batch_y)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    return {"loss": avg_loss, "f1_macro": f1, "balanced_acc": bal_acc}


# ==============================================================================
# GESTOR DE CHECKPOINTS
# ==============================================================================

class CheckpointManager:
    """
    Gestiona el guardado y carga de checkpoints de forma atómica y segura.

    Por qué atómico:
      Si el proceso muere a mitad del guardado (SIGKILL, fallo de disco),
      un checkpoint corrupto es peor que ninguno. Guardamos en un archivo
      temporal y luego hacemos rename atómico (operación del SO).

    Estructura de directorio:
      checkpoints/
        ├── best_model.pt          ← Mejor modelo por val F1
        ├── checkpoint_epoch_05.pt ← Checkpoint cada N epochs
        ├── checkpoint_epoch_10.pt
        ├── checkpoint_latest.pt   ← Siempre el más reciente (symlink)
        └── training_state.json    ← Metadatos legibles
    """

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._saved_checkpoints = []

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer,
        scheduler,
        metrics: Dict,
        config: TrainingConfig,
        is_best: bool = False,
        tag: str = "",
    ) -> Path:
        """
        Guarda checkpoint de forma atómica.

        Returns:
            Path del checkpoint guardado
        """
        # En DDP, solo el proceso rank 0 guarda (evitar escrituras concurrentes)
        if dist.is_initialized() and dist.get_rank() != 0:
            return None

        filename = f"checkpoint_epoch_{epoch:04d}{('_' + tag) if tag else ''}.pt"
        tmp_path  = self.checkpoint_dir / f"._tmp_{filename}"
        final_path = self.checkpoint_dir / filename

        # Extraer state_dict del modelo (sin wrapper DDP)
        model_state = (
            model.module.state_dict()
            if isinstance(model, DDP) else model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": vars(config) if hasattr(config, "__dict__") else {},
            "timestamp": datetime.now().isoformat(),
            "torch_version": torch.__version__,
        }

        # ── Guardado atómico ───────────────────────────────────────────────────
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(final_path)  # rename() es atómico en sistemas POSIX
        print(f"[Checkpoint] Guardado: {final_path}")

        # ── Guardar metadatos JSON legibles ───────────────────────────────────
        meta = {
            "last_checkpoint": str(final_path),
            "epoch": epoch,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "timestamp": checkpoint["timestamp"],
        }
        meta_path = self.checkpoint_dir / "training_state.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # ── Actualizar symlink "latest" ────────────────────────────────────────
        latest_link = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # En sistemas sin symlinks: copiar el archivo
        try:
            latest_link.symlink_to(final_path.name)
        except (OSError, NotImplementedError):
            shutil.copy2(final_path, latest_link)

        # ── Guardar mejor modelo por separado ────────────────────────────────
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy2(final_path, best_path)
            print(f"[Checkpoint] ★ Nuevo mejor modelo guardado: {best_path}")

        # ── Rotar checkpoints antiguos (conservar keep_last_n) ────────────────
        self._saved_checkpoints.append(final_path)
        if len(self._saved_checkpoints) > self.keep_last_n:
            old = self._saved_checkpoints.pop(0)
            if old.exists() and "best" not in old.name:
                old.unlink()
                print(f"[Checkpoint] Eliminado checkpoint antiguo: {old.name}")

        return final_path

    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer=None,
        scheduler=None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Carga un checkpoint. Soporta 'latest' y 'best' como atajos.

        Returns:
            dict con epoch, metrics, y otros metadatos del checkpoint
        """
        if path == "latest":
            ckpt_path = self.checkpoint_dir / "checkpoint_latest.pt"
        elif path == "best":
            ckpt_path = self.checkpoint_dir / "best_model.pt"
        else:
            ckpt_path = Path(path)

        if not ckpt_path.exists():
            print(f"[Checkpoint] No se encontró checkpoint en: {ckpt_path}")
            return {}

        print(f"[Checkpoint] Cargando desde: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Cargar pesos del modelo
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        # Restaurar estado del optimizador y scheduler
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch   = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        ts      = checkpoint.get("timestamp", "desconocido")

        print(f"  → Epoch: {epoch} | Métricas: {metrics} | Guardado: {ts}")

        return {"epoch": epoch, "metrics": metrics}

    def find_latest(self) -> Optional[Path]:
        """Busca el checkpoint más reciente en el directorio."""
        candidates = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None


# ==============================================================================
# MANEJADOR DE SEÑALES (SIGTERM / SIGUSR1)
# ==============================================================================

class GracefulKiller:
    """
    Intercepta señales del sistema para guardar checkpoint antes de morir.

    Señales relevantes en servidores compartidos:
      SIGTERM: enviada por 'docker stop', 'kill <pid>', o scheduler SLURM
      SIGUSR1: señal personalizable para "aviso de parada próxima"
      SIGINT:  Ctrl+C del usuario
    """

    def __init__(self):
        self.kill_now = False
        self._checkpoint_fn = None

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGUSR1, self._handle_checkpoint_signal)

    def register_checkpoint_fn(self, fn):
        """Registra la función que guarda el checkpoint de emergencia."""
        self._checkpoint_fn = fn

    def _handle_signal(self, signum, frame):
        sig_name = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}.get(signum, str(signum))
        print(f"\n[SEÑAL] Recibida {sig_name}. Guardando checkpoint de emergencia...")

        if self._checkpoint_fn is not None:
            try:
                self._checkpoint_fn(tag="emergency")
                print("[SEÑAL] Checkpoint de emergencia guardado. Terminando.")
            except Exception as e:
                print(f"[SEÑAL] Error al guardar checkpoint: {e}")

        self.kill_now = True

    def _handle_checkpoint_signal(self, signum, frame):
        """SIGUSR1: guardar checkpoint sin terminar (útil para snapshots manuales)."""
        print("[SEÑAL] SIGUSR1 recibida — guardando snapshot manual...")
        if self._checkpoint_fn is not None:
            try:
                self._checkpoint_fn(tag="manual")
                print("[SEÑAL] Snapshot manual guardado.")
            except Exception as e:
                print(f"[SEÑAL] Error al guardar snapshot: {e}")


# ==============================================================================
# SETUP DDP
# ==============================================================================

def setup_ddp():
    """
    Inicializa el grupo de procesos distribuidos para DDP.

    torchrun configura automáticamente las variables de entorno:
      RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Backend NCCL: óptimo para GPU-GPU communication (NVLink o PCIe)
    # Gloo: alternativa para CPU o cuando NCCL falla
    from datetime import timedelta
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=10),   # falla rápido si hay deadlock
        )



    # Cada proceso usa su GPU asignada
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"[DDP] Inicializado — Rank: {rank}/{world_size-1} | GPU: {local_rank}")
        print(f"[DDP] Backend: NCCL | Master: {os.environ.get('MASTER_ADDR')}:"
              f"{os.environ.get('MASTER_PORT')}")

    return rank, local_rank, world_size, device


def cleanup_ddp():
    """Limpia el grupo de procesos al finalizar."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ==============================================================================
# DATALOADER DISTRIBUIDO
# ==============================================================================

def build_distributed_dataloaders(
    train_pnpl, val_pnpl, test_pnpl,
    preprocessor: MEGPreprocessor,
    img_converter: MEGToImage,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
):
    """
    Construye DataLoaders con DistributedSampler para DDP.

    Con DDP + DistributedSampler:
      - Cada GPU procesa un subconjunto distinto del dataset
      - El batch_size aquí es POR GPU (el efectivo global = batch_size × world_size)
      - Ejemplo: batch_size=32, 2 GPUs → batch global efectivo = 64

    IMPORTANTE: En DDP, el sampler maneja el shuffle (no el DataLoader).
    """

    train_ds = MEGImageDataset(train_pnpl, preprocessor, img_converter, augment=True)
    val_ds   = MEGImageDataset(val_pnpl,   preprocessor, img_converter, augment=False)
    test_ds  = MEGImageDataset(test_pnpl,  preprocessor, img_converter, augment=False)

    # DistributedSampler: divide el dataset entre los procesos
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,   # Importante para que todos los ranks tengan mismo nº batches
    )
    # Validación y test: cada rank evalúa TODO el dataset (luego se agregan métricas)
    val_sampler  = DistributedSampler(val_ds,  num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=3,      # <-- añadir
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=3,      # <-- añadir
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=3,      # <-- añadir
    )

    

    return train_loader, val_loader, test_loader, train_sampler


# ==============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO DDP
# ==============================================================================

def train_ddp(args):
    """
    Bucle de entrenamiento distribuido (DDP) con:
      - Checkpointing automático cada N epochs
      - Guardado de emergencia ante SIGTERM
      - Resume desde último checkpoint
      - Logging en TensorBoard (solo rank 0)
    """
    # Al inicio de train_ddp(), ANTES de setup_ddp()
    if int(os.environ.get("RANK", 0)) == 0:
        print("[PRE] Precalculando stats H5 (rank 0 solo)...")
        load_libribrain(LibriBrainConfig(args.data_path, args.task, "train"))
        load_libribrain(LibriBrainConfig(args.data_path, args.task, "validation"))
        load_libribrain(LibriBrainConfig(args.data_path, args.task, "test"))
        print("[PRE] Stats calculadas. Iniciando DDP...")

    # AHORA iniciar DDP (todos los procesos llegan aquí)
    rank, local_rank, world_size, device = setup_ddp()
    dist.barrier()  # Asegurar que rank 0 terminó antes de continuar

    # ── Setup DDP ─────────────────────────────────────────────────────────────
    rank, local_rank, world_size, device = setup_ddp()
    is_main = (rank == 0)  # Solo el proceso 0 imprime y guarda

    # ── Manejador de señales (todos los procesos) ─────────────────────────────
    killer = GracefulKiller()

    # ── TensorBoard (solo rank 0) ──────────────────────────────────────────────
    writer = None
    if is_main:
        log_dir = Path(args.output_dir) / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[TensorBoard] Logs en: {log_dir}")
        print(f"              Ver con: tensorboard --logdir {args.output_dir}/tensorboard")

    # ── Checkpoint Manager ────────────────────────────────────────────────────
    ckpt_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_last_n=3,
    )

    # ── Cargar datos ──────────────────────────────────────────────────────────
    if is_main:
        print(f"\n[PASO 1] Cargando LibriBrain — tarea: {args.task}")

    # Rank 0 primero: calcula y escribe las stats en los H5 (modo "r+")
    if rank == 0:
        train_pnpl, n_classes, n_channels = load_libribrain(
            LibriBrainConfig(args.data_path, args.task, "train")
        )
        val_pnpl,  _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "validation"))
        test_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "test"))

    # Esperar a que rank 0 termine de escribir antes de que los demás abran los mismos archivos
    dist.barrier()

    # El resto de ranks cargan ya con las stats cacheadas (solo lectura)
    if rank != 0:
        train_pnpl, n_classes, n_channels = load_libribrain(
            LibriBrainConfig(args.data_path, args.task, "train")
        )
        val_pnpl,  _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "validation"))
        test_pnpl, _, _ = load_libribrain(LibriBrainConfig(args.data_path, args.task, "test"))

    # ── Preprocesado y conversión imagen ─────────────────────────────────────
    preprocessor  = MEGPreprocessor(use_instance_norm=True, clip_std=5.0)
    img_converter = MEGToImage(sfreq=250.0, n_freqs=args.n_freqs, img_size=224)

    # ── DataLoaders distribuidos ──────────────────────────────────────────────
    if is_main:
        print(f"\n[PASO 2] Construyendo DataLoaders distribuidos")
        print(f"  Batch por GPU: {args.batch_size} | Batch global: {args.batch_size * world_size}")

    train_loader, val_loader, test_loader, train_sampler = build_raw_dataloaders(
        train_pnpl, val_pnpl, test_pnpl,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=True,
        rank=rank,
        world_size=dist.get_world_size(),
    )

        # ── Pesos de clase ────────────────────────────────────────────────────────
    # Lectura directa de los TSV de eventos (rápido, sin tocar los H5)
    if rank == 0:
        print("[INFO] Calculando pesos de clase desde TSVs de eventos...", flush=True)
        import glob
        import pandas as pd
        
        # Buscar todos los TSV de eventos del split de training
        # Los TSVs están en: libribrain_data/Sherlock*/derivatives/events/*.tsv
        tsv_files = sorted(glob.glob(
            f"{args.data_path}/Sherlock*/derivatives/events/*_events.tsv"
        ))
        print(f"[INFO] Encontrados {len(tsv_files)} archivos TSV de eventos", flush=True)
        
        # Cargar solo las filas de phonemes
        all_phonemes = []
        for tsv in tsv_files:
            try:
                df = pd.read_csv(tsv, sep='\t')
                # Filtrar solo eventos de tipo 'phoneme' (no 'word', 'silence', etc.)
                phoneme_rows = df[df['type'] == 'phoneme'] if 'type' in df.columns else df
                if 'value' in phoneme_rows.columns:
                    all_phonemes.extend(phoneme_rows['value'].tolist())
            except Exception as e:
                print(f"  [WARN] Error leyendo {tsv}: {e}", flush=True)
        
        print(f"[INFO] Total fonemas en TSVs: {len(all_phonemes)}", flush=True)
        
        # Mapear strings a índices usando el mapping del propio dataset
        if hasattr(train_pnpl, 'phoneme_to_id'):
            ph_map = train_pnpl.phoneme_to_id
        elif hasattr(train_pnpl, 'label_map'):
            ph_map = train_pnpl.label_map
        else:
            # Fallback: contar strings únicos
            unique = sorted(set(all_phonemes))
            ph_map = {p: i for i, p in enumerate(unique)}
        
        # Contar ocurrencias por clase
        counts = np.zeros(n_classes, dtype=np.float64)
        for ph in all_phonemes:
            if ph in ph_map and ph_map[ph] < n_classes:
                counts[ph_map[ph]] += 1
        
        # Si la lectura de TSVs falló o dio cero, usar pesos uniformes
        if counts.sum() == 0:
            print("[WARN] No se pudieron leer labels de TSVs. Usando pesos uniformes.", flush=True)
            class_weights = torch.ones(n_classes, device=device) / n_classes
        else:
            # Inverse Number of Samples (ISNS) — lo mismo que tu compute_class_weights_isns
            counts = np.maximum(counts, 1)  # evitar division por cero
            weights = 1.0 / np.sqrt(counts)
            weights = weights / weights.sum() * n_classes  # normalizar
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        print(f"[INFO] Pesos: min={class_weights.min():.4f} max={class_weights.max():.4f}", flush=True)
    else:
        class_weights = torch.zeros(n_classes, device=device)

    # Broadcast a todos los ranks
    dist.broadcast(class_weights, src=0)
    dist.barrier()  # asegurar que todos los ranks tienen los pesos antes de continuar

    # ── Modelo ────────────────────────────────────────────────────────────────
    if is_main:
        print(f"\n[PASO 3] Construyendo modelo: {args.backbone} | {args.strategy}")

    if rank == 0:
        if is_main:
            print(f"\n[PASO 3] Construyendo modelo: {args.backbone} | {args.strategy}")
        # Forzar descarga (no cuesta nada si ya está cacheado)
        import torchvision.models as tvm
        if args.pretrained:
            if args.backbone == "resnet18":
                _ = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            # añade aquí los demás backbones si los usas
            elif args.backbone == "resnet50":
                _ = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            elif args.backbone == "resnet101":
                _ = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V1)  
            elif args.backbone == "resnet152":
                _ = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V1)
            elif args.backbone == "efficientnet_b0":
                _ = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif args.backbone == "efficientnet_b1":
                _ = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.IMAGENET1K_V1)
            elif args.backbone == "efficientnet_b2":
                _ = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.IMAGENET1K_V1)
            else:
                print(f"[WARN] Backbone {args.backbone} no reconocido para precarga. "
                      f"Revisa el código si usas un backbone distinto y añádelo vago.")
            
    dist.barrier()  # los demás ranks esperan a que rank 0 termine la descarga

    cwt_layer = CWTLayer(
        sfreq=250.0, n_freqs=96, f_min=1.0, f_max=125.0, B=1.5, C=1.0).to(device)

    model = MEGImageModelEndToEnd(
        backbone_name=args.backbone,
        n_classes=n_classes,
        n_meg_channels=306,
        n_freqs=96,
        img_size=224,
        pretrained=True,
        strategy=args.strategy,
    ).to(device)
    model = DDP(model, device_ids=[local_rank])

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Parámetros: {trainable:,} entrenables / {total:,} totales")

    # ── Configuración de entrenamiento ────────────────────────────────────────
    config = TrainingConfig(
        backbone=args.backbone,
        strategy=args.strategy,
        n_classes=n_classes,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # ── Optimizador ───────────────────────────────────────────────────────────
    # Acceder a model.module para los param groups (DDP envuelve el modelo)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model.module, config, len(train_loader)
    )

    # ── Loss con pesos de clase ───────────────────────────────────────────────
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Resume desde checkpoint ───────────────────────────────────────────────
    start_epoch = 1
    best_val_f1 = 0.0

    if args.resume_from and args.resume_from != "none":
        ckpt_info = ckpt_manager.load(
            args.resume_from, model, optimizer, scheduler, device
        )
        if ckpt_info:
            start_epoch = ckpt_info.get("epoch", 0) + 1
            best_val_f1 = ckpt_info.get("metrics", {}).get("f1_macro", 0.0)
            if is_main:
                print(f"[Resume] Continuando desde epoch {start_epoch} | Mejor F1: {best_val_f1:.4f}")

    # ── Registrar función de checkpoint de emergencia en killer ───────────────
    def emergency_checkpoint(tag="emergency"):
        ckpt_manager.save(
            epoch=getattr(train_ddp, "_current_epoch", 0),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={"f1_macro": best_val_f1},
            config=config,
            is_best=False,
            tag=tag,
        )

    killer.register_checkpoint_fn(emergency_checkpoint)

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(patience=config.patience, mode="max")

    # ── Bucle de entrenamiento ────────────────────────────────────────────────
    if is_main:
        print(f"\n{'='*60}")
        print(f"  INICIO ENTRENAMIENTO DDP")
        print(f"  GPUs: {world_size} × RTX 6000")
        print(f"  Epochs: {start_epoch} → {args.n_epochs}")
        print(f"  Checkpoint cada: {args.checkpoint_every} epochs")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.n_epochs + 1):
        train_ddp._current_epoch = epoch  # Para el checkpoint de emergencia

        # ── Verificar señal de parada ─────────────────────────────────────────
        if killer.kill_now:
            if is_main:
                print(f"[SEÑAL] Parando entrenamiento en epoch {epoch}.")
            break

        # ── Sincronizar epoch entre procesos (DistributedSampler requiere esto)
        train_sampler.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────────
        t0 = time.time()
        train_metrics = train_one_epoch_raw(
            model, cwt_layer, train_loader, optimizer, criterion, device,
            grad_clip=config.grad_clip,
        )
        train_time = time.time() - t0

        # ── Val (todos los ranks evalúan, rank 0 reporta) ─────────────────────
        val_metrics = evaluate_raw(model, cwt_layer, val_loader, criterion, device)

        # ── Agregar métricas de validación entre ranks ────────────────────────
        # DDP necesita agregar métricas de todos los ranks
        val_f1_tensor = torch.tensor(val_metrics["f1_macro"], device=device)
        dist.all_reduce(val_f1_tensor, op=dist.ReduceOp.AVG)
        val_metrics["f1_macro"] = val_f1_tensor.item()

        # ── Scheduler ─────────────────────────────────────────────────────────
        if scheduler:
            scheduler.step()

        # ── Logging (solo rank 0) ─────────────────────────────────────────────
        if is_main:
            is_best = val_metrics["f1_macro"] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics["f1_macro"]

            print(
                f"Epoch {epoch:04d}/{args.n_epochs} │ "
                f"Train Loss: {train_metrics['loss']:.4f} │ "
                f"Train F1: {train_metrics['f1_macro']:.4f} │ "
                f"Val F1: {val_metrics['f1_macro']:.4f} "
                f"{'★ BEST' if is_best else ''} │ "
                f"Tiempo: {train_time:.1f}s"
            )

            # TensorBoard
            if writer:
                writer.add_scalar("Train/Loss",        train_metrics["loss"],        epoch)
                writer.add_scalar("Train/F1_macro",    train_metrics["f1_macro"],    epoch)
                writer.add_scalar("Val/F1_macro",      val_metrics["f1_macro"],      epoch)
                writer.add_scalar("Val/Balanced_Acc",  val_metrics["balanced_acc"],  epoch)
                writer.add_scalar("LR/head",
                                  optimizer.param_groups[0]["lr"], epoch)

            # ── Checkpoint periódico ───────────────────────────────────────────
            if epoch % args.checkpoint_every == 0 or is_best or killer.kill_now:
                ckpt_manager.save(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    config=config,
                    is_best=is_best,
                )

        # ── Sincronizar entre procesos antes de continuar ─────────────────────
        dist.barrier()

        # ── Early stopping (decidido por rank 0, comunicado a todos) ──────────
        stop_tensor = torch.zeros(1, device=device)
        if is_main:
            should_stop = early_stopper.step(val_metrics["f1_macro"], model.module)
            if should_stop:
                stop_tensor[0] = 1.0
                print(f"\n[Early Stopping] Activado en epoch {epoch}. "
                      f"Mejor F1: {early_stopper.best_value:.4f}")
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item() > 0:
            # Restaurar mejor modelo en todos los ranks
            if is_main:
                early_stopper.restore_best(model.module)
            break

    # ── Evaluación final en test ───────────────────────────────────────────────
    # Cargar best checkpoint en TODOS los ranks antes del test eval
    best_path = Path(args.checkpoint_dir) / "best_model.pt"
    if best_path.exists():
        if is_main:
            print(f"\n[EVALUACIÓN FINAL] Cargando best checkpoint en todos los ranks...")
        ckpt_manager.load("best", model, device=device)

    # Sincronizar antes del test eval
    dist.barrier()

    test_metrics = evaluate_raw(model, cwt_layer, test_loader, criterion, device)

    # Agregar métricas de test entre GPUs
    for key in ["f1_macro", "balanced_acc"]:
        t = torch.tensor(test_metrics[key], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        test_metrics[key] = t.item()

    if is_main:
        print(f"\n{'='*50}")
        print(f"  RESULTADOS FINALES (TEST SET)")
        print(f"  F1-macro:      {test_metrics['f1_macro']:.4f}")
        print(f"  Balanced Acc:  {test_metrics['balanced_acc']:.4f}")
        print(f"{'='*50}")

        # Guardar resultados finales
        results_path = Path(args.output_dir) / "final_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "backbone": args.backbone,
                "strategy": args.strategy,
                "task": args.task,
                "test_f1_macro": test_metrics["f1_macro"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "best_val_f1": best_val_f1,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Resultados guardados: {results_path}")

        if writer:
            writer.close()

    # ── Cleanup DDP ───────────────────────────────────────────────────────────
    cleanup_ddp()


# ==============================================================================
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MEG Transfer Learning — Entrenamiento DDP multi-GPU"
    )
    # Tarea y modelo
    parser.add_argument("--task",      default="phoneme", choices=["speech", "phoneme"])
    parser.add_argument("--backbone",  default="resnet18",
                        choices=["resnet18", "efficientnet_b0", "vit_tiny", "vit_base"])
    parser.add_argument("--strategy",  default="partial_ft",
                        choices=["frozen", "partial_ft", "full_ft"])
    parser.add_argument("--pretrained",    action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Entrenamiento
    parser.add_argument("--n_epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=32,  help="Batch POR GPU")
    parser.add_argument("--n_freqs",         type=int,   default=96)
    parser.add_argument("--num_workers",     type=int,   default=4)

    # Checkpointing
    parser.add_argument("--checkpoint_dir",   default="./checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=1,
                        help="Guardar checkpoint cada N epochs (1 = cada epoch)")
    parser.add_argument("--resume_from",      default="none",
                        help="'latest', 'best', o path a .pt. 'none' para empezar de cero")

    # Paths
    parser.add_argument("--data_path",  default="./libribrain_data")
    parser.add_argument("--output_dir", default="./results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ddp(args)
