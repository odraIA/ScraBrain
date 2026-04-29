"""
================================================================================
  Transfer Learning desde ImageNet para Clasificación MEG en LibriBrain
================================================================================

DESCRIPCIÓN GENERAL
-------------------
Este script implementa el pipeline completo para decodificar señales MEG del
dataset LibriBrain usando representaciones de imágenes tiempo-frecuencia y
transfer learning desde modelos preentrenados en ImageNet.

Basado en:
  - Jhilal et al. (ISBI 2026): "Transfer Learning from ImageNet for MEG-based
    Decoding of Imagined Speech" → pipeline de imagen (CWT + conv 1×1 + ResNet/ViT)
  - LibriBrain (Özdogan et al.): dataset con 306 canales, 250 Hz, tareas de
    Speech Detection (binaria) y Phoneme Classification (39 clases)
  - MEGConformer (de Zuazo et al., NeurIPS 2025): instance norm + data augment
  - MEBM-Phoneme (NeurIPS 2025): multiscale conv + atención temporal

TAREAS SOPORTADAS
-----------------
  1. Speech Detection     : binaria (habla vs silencio)
  2. Phoneme Classification: 39 clases de fonemas

ESTRATEGIAS DE ENTRENAMIENTO
-----------------------------
  A) Transfer Learning puro   → backbone completamente congelado, solo se entrena la cabeza
  B) Fine-tuning parcial      → último bloque + cabeza desentrenados (MEJOR según ablación ISBI)
  C) Fine-tuning completo     → todos los pesos actualizados (riesgo de overfitting)

MODELOS DISPONIBLES
-------------------
  - ResNet-18    (ganador en ISBI 2026)
  - EfficientNet-B0 (eficiente y competitivo)
  - ViT-Tiny     (transformer de visión, segundo mejor en ISBI 2026)
  - ViT-B/16     (versión más grande de ViT)

DEPENDENCIAS
------------
    pip install pnpl torch torchvision torchaudio timm pywavelets \
                scikit-learn matplotlib numpy scipy tqdm

ESTRUCTURA DEL SCRIPT
---------------------
  SECCIÓN 0: Imports y configuración global
  SECCIÓN 1: Carga de datos con pnpl (LibriBrain)
  SECCIÓN 2: Preprocesado de señales MEG
  SECCIÓN 3: Creación de representaciones imagen (CWT → escalograma → imagen RGB)
  SECCIÓN 4: Dataset PyTorch con generación de imágenes on-the-fly
  SECCIÓN 5: Definición de modelos con proyección sensor-espacio
  SECCIÓN 6: Estrategias de transfer learning y fine-tuning
  SECCIÓN 7: Bucle de entrenamiento con métricas y early stopping
  SECCIÓN 8: Evaluación y comparativa de estrategias
  SECCIÓN 9: Script principal (main) con experimentos completos

REFERENCIA RÁPIDA
-----------------
  Para ejecutar todos los experimentos:
      python meg_transfer_learning_libribrain.py

  Para ejecutar solo una tarea con un modelo específico:
      python meg_transfer_learning_libribrain.py \
          --task phoneme --model resnet18 --strategy partial_ft

================================================================================
"""

# ==============================================================================
# SECCIÓN 0: IMPORTS Y CONFIGURACIÓN GLOBAL
# ==============================================================================

import os
import json
import argparse
import csv
from unittest import loader
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

try:
    import timm  # Para ViT-Tiny y ViT-B/16
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    warnings.warn("timm no instalado. ViT no disponible. Instalar con: pip install timm")

import pywt  # PyWavelets para la Transformada Wavelet Continua (CWT)
from scipy import signal as scipy_signal
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Cargar LibriBrain mediante pnpl
try:
    from pnpl.datasets import LibriBrainSpeech, LibriBrainPhoneme
    PNPL_AVAILABLE = True
except ImportError:
    PNPL_AVAILABLE = False
    warnings.warn(
        "pnpl no instalado o no disponible. Instalar con:\n"
        "  pip install pnpl\n"
        "  o bien: pip install git+https://github.com/neural-processing-lab/frozen-pnpl"
    )

warnings.filterwarnings("ignore")

# ─── Reproducibilidad ────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo: {DEVICE}")


# ==============================================================================
# SECCIÓN 1: CARGA DE DATOS CON pnpl (LibriBrain)
# ==============================================================================

@dataclass
class LibriBrainConfig:
    """
    Configuración del dataset LibriBrain.

    Características del dataset:
      - 306 canales MEG (magnetómetros + gradiantes del sistema MEGIN Triux Neo)
      - Muestreado a 250 Hz (4 ms por muestra) tras downsampling desde 1 kHz
      - Sujeto único escuchando >50h de audiolibros (Sherlock Holmes)
      - 91 sesiones de entrenamiento, 1 de validación, 1 de test
      - Tareas: Speech Detection (binaria) y Phoneme Classification (39 clases)
    """
    data_path: str = "./libribrain_data"
    task: str = "phoneme"          # "speech" | "phoneme"
    partition: str = "train"       # "train" | "validation" | "test"
    download: bool = True


def load_libribrain(config: LibriBrainConfig):
    """
    Carga el dataset LibriBrain usando la librería pnpl.

    El dataloader pnpl devuelve epochs ya pre-segmentadas y etiquetadas.
    Cada sample es un array de forma (306, T) donde:
      - 306 = número de canales MEG
      - T   = número de muestras temporales en la ventana

    Para Speech Detection:
      - Ventanas de 2.5 s → T ≈ 625 muestras a 250 Hz
      - Etiquetas: 0 = silencio, 1 = habla

    Para Phoneme Classification:
      - Ventanas centradas en cada fonema
      - Etiquetas: 0-38 (39 fonemas del ARPAbet)
      - En el holdout los samples están promediados sobre 100 repeticiones para ↑ SNR

    Returns:
        dataset: objeto Dataset de pnpl iterable
        n_classes: número de clases para la tarea elegida
        n_channels: número de canales MEG (306 para LibriBrain)
    """
    if not PNPL_AVAILABLE:
        raise RuntimeError(
            "pnpl no está disponible. Instalar con:\n"
            "  pip install git+https://github.com/neural-processing-lab/frozen-pnpl"
        )

    Path(config.data_path).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Cargando LibriBrain – tarea: {config.task}, partición: {config.partition}")

    if config.task == "speech":
        dataset = LibriBrainSpeech(
            data_path=config.data_path,
            partition=config.partition,
            download=config.download,
        )
        n_classes = 2
    elif config.task == "phoneme":
        dataset = LibriBrainPhoneme(
            data_path=config.data_path,
            partition=config.partition,
            download=config.download,
        )
        n_classes = 39
    else:
        raise ValueError(f"Tarea desconocida: {config.task}. Usar 'speech' o 'phoneme'.")

    n_channels = 306  # Fijo en LibriBrain

    print(f"[INFO] Dataset cargado: {len(dataset)} samples, {n_classes} clases, {n_channels} canales")
    return dataset, n_classes, n_channels


# ==============================================================================
# SECCIÓN 2: PREPROCESADO DE SEÑALES MEG
# ==============================================================================

class MEGPreprocessor:
    """
    Preprocesado de señales MEG para el pipeline de imagen.

    Pipeline aplicado:
    1. Instance Normalization por canal (crítico para generalización en holdout,
       según MEGConformer: cierra la brecha de distribución entre sesiones)
    2. Z-score por canal relativo a la baseline (como en Jhilal et al. 2026)
    3. Clip de outliers (±5 desviaciones estándar) para robustez

    NOTA: LibriBrain ya viene con preprocesado mínimo aplicado:
      - Corrección de movimiento de cabeza
      - Filtrado de ruido (notch 50 Hz y bandpass)
      - Downsampling a 250 Hz
    Aquí solo aplicamos normalización a nivel de epoch.
    """

    def __init__(
        self,
        use_instance_norm: bool = True,   # Recomendado para robustez en holdout
        baseline_samples: Optional[int] = None,  # Número de muestras de baseline
        clip_std: float = 5.0,
    ):
        self.use_instance_norm = use_instance_norm
        self.baseline_samples = baseline_samples
        self.clip_std = clip_std

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        """
        Args:
            epoch: array (n_channels, n_times) con señal MEG cruda

        Returns:
            epoch_proc: array (n_channels, n_times) preprocesado
        """
        epoch = epoch.astype(np.float32)

        # ── Paso 1: Instance Normalization por canal ──────────────────────────
        # Normaliza cada canal individualmente (sin estadísticas running).
        # Elimina drift de amplitud entre ventanas y sesiones.
        # Según MEGConformer: mejora >200% en generalización al holdout.
        if self.use_instance_norm:
            mean = epoch.mean(axis=1, keepdims=True)
            std  = epoch.std(axis=1, keepdims=True) + 1e-8
            epoch = (epoch - mean) / std

        # ── Paso 2: Z-score relativo a baseline ───────────────────────────────
        # Si se especifica baseline, normalizar respecto a pre-estímulo.
        # Útil cuando las epochs tienen un período de pre-cue (-300 ms como
        # en el paper ISBI 2026) o cuando se dispone de señal de reposo.
        if self.baseline_samples is not None and self.baseline_samples > 0:
            baseline = epoch[:, :self.baseline_samples]
            b_mean   = baseline.mean(axis=1, keepdims=True)
            b_std    = baseline.std(axis=1, keepdims=True) + 1e-8
            epoch    = (epoch - b_mean) / b_std

        # ── Paso 3: Clip de outliers ──────────────────────────────────────────
        # Elimina artefactos residuales (parpadeos, movimientos)
        if self.clip_std > 0:
            epoch = np.clip(epoch, -self.clip_std, self.clip_std)

        return epoch


class LinearSourceProjector:
    """
    Aplica una proyección lineal antes de la CWT: source = W @ sensors.

    La idea es mantener el entrenamiento desacoplado de la construcción del
    modelo directo/inverso. W puede venir de MNE-Python (LCMV por vértice o ROI),
    DSS, PCA anatómicamente agregada, etc. El contrato es:

        W.shape == (n_sources_or_rois, n_sensor_channels)
        epoch.shape == (n_sensor_channels, n_times)

    Formatos soportados:
      - .npy: array W
      - .npz: clave "W", "filters" o "projection"
      - .pt/.pth: tensor o dict con "W", "filters" o "projection"
      - .json: lista 2D
    """

    def __init__(self, projection_path: str, name: str = "source"):
        self.projection_path = projection_path
        self.name = name
        self.matrix = self._load_matrix(projection_path)
        if self.matrix.ndim != 2:
            raise ValueError(
                f"La matriz de proyección debe ser 2D; recibido {self.matrix.shape}"
            )
        self.n_outputs, self.n_inputs = self.matrix.shape

    @staticmethod
    def _load_matrix(path: str) -> np.ndarray:
        suffix = Path(path).suffix.lower()

        if suffix == ".npy":
            matrix = np.load(path)
        elif suffix == ".npz":
            data = np.load(path)
            for key in ("W", "filters", "projection"):
                if key in data:
                    matrix = data[key]
                    break
            else:
                raise KeyError(
                    f"{path} debe contener una clave W, filters o projection"
                )
        elif suffix in (".pt", ".pth"):
            data = torch.load(path, map_location="cpu")
            if isinstance(data, dict):
                for key in ("W", "filters", "projection"):
                    if key in data:
                        data = data[key]
                        break
                else:
                    raise KeyError(
                        f"{path} debe contener una clave W, filters o projection"
                    )
            matrix = data.detach().cpu().numpy() if torch.is_tensor(data) else data
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                matrix = json.load(f)
        else:
            raise ValueError(
                f"Formato no soportado para source_projection_path: {suffix}. "
                "Usa .npy, .npz, .pt/.pth o .json."
            )

        return np.asarray(matrix, dtype=np.float32)

    def validate_input_channels(self, n_channels: int):
        if self.n_inputs != n_channels:
            raise ValueError(
                f"La proyección {self.projection_path} espera {self.n_inputs} "
                f"canales, pero LibriBrain entregó {n_channels}."
            )

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        if epoch.shape[0] != self.n_inputs:
            raise ValueError(
                f"Epoch con {epoch.shape[0]} canales; la proyección espera "
                f"{self.n_inputs}."
            )
        return (self.matrix @ epoch).astype(np.float32)


# ==============================================================================
# SECCIÓN 3: CREACIÓN DE REPRESENTACIONES IMAGEN (TF → RGB)
# ==============================================================================

class MEGToImage:
    """
    Convierte epochs MEG en imágenes 224×224×3 compatibles con modelos ImageNet.

    Pipeline (siguiendo Jhilal et al., ISBI 2026):
    ┌─────────────────────┐
    │  Epoch MEG          │  (306 canales × T muestras)
    │  (306, T)           │
    └──────────┬──────────┘
               │ CWT por canal (Morlet, 96 frecuencias log)
               ▼
    ┌─────────────────────┐
    │  Scalograms         │  (306, 96 freq, T)
    │  (306, F, T)        │
    └──────────┬──────────┘
               │ Z-score relativo a baseline por frecuencia
               ▼
    ┌─────────────────────┐
    │  Conv 1×1 learnable │  Mezcla de canales: 306 → 3
    │  (o PCA-3 fijo)     │  (aprendida durante el entrenamiento)
    └──────────┬──────────┘
               │ Reshape + resize bilineal
               ▼
    ┌─────────────────────┐
    │  Imagen RGB         │  (3, 224, 224)
    │  lista para ResNet  │
    └─────────────────────┘

    Parámetros clave:
      - n_freqs   : número de bins de frecuencia (96, log-espaciados de 1 a 125 Hz)
      - f_min/max : rango de frecuencias (1-125 Hz, hasta Nyquist de 250 Hz)
      - img_size  : tamaño final de imagen (224×224 para ImageNet)
      - wavelet   : 'cmor1.5-1.0' (Morlet complejo) como en el paper ISBI
    """

    def __init__(
        self,
        sfreq: float = 250.0,           # Frecuencia de muestreo LibriBrain
        n_freqs: int = 96,              # Bins de frecuencia (igual que ISBI 2026)
        f_min: float = 1.0,             # Frecuencia mínima (Hz)
        f_max: float = 125.0,           # Frecuencia máxima (Hz, Nyquist = 125)
        img_size: int = 224,            # Tamaño imagen destino (ImageNet standard)
        wavelet: str = "cmor1.5-1.0",   # Morlet complejo (igual que ISBI 2026)
        projection: str = "pca",        # "pca" (fija) | "learned" (requiere nn.Conv2d externo)
        n_pca_components: int = 3,
    ):
        self.sfreq = sfreq
        self.n_freqs = n_freqs
        self.img_size = img_size
        self.wavelet = wavelet
        self.projection = projection
        self.n_pca_components = n_pca_components

        # Escala de frecuencias log-espaciadas (igual que en ISBI 2026)
        self.frequencies = np.logspace(
            np.log10(f_min), np.log10(f_max), n_freqs
        )
        # Convertir frecuencias a escalas para pywt
        self.scales = pywt.frequency2scale(wavelet, self.frequencies / sfreq)

        # Si se usa PCA para proyectar los canales: se ajusta al primer batch
        self.pca_fitted = False
        self.pca_components = None  # shape: (3, 306)

    def compute_cwt_channel(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        Calcula el escalograma CWT para un canal MEG.

        Args:
            signal_1d: array (T,) con la señal de un canal

        Returns:
            scalogram: array (n_freqs, T) con |coeficientes de Morlet|
        """
        # CWT de Morlet → coeficientes complejos (n_freqs, T)
        coeffs, _ = pywt.cwt(signal_1d, self.scales, self.wavelet)
        # Magnitud del coeficiente = potencia espectro-temporal
        scalogram = np.abs(coeffs).astype(np.float32)  # (n_freqs, T)
        return scalogram

    def compute_all_scalograms(self, epoch: np.ndarray) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor
        n_channels, T = epoch.shape
        scalograms = np.zeros((n_channels, self.n_freqs, T), dtype=np.float32)

        def _cwt_ch(ch):
            scalograms[ch] = self.compute_cwt_channel(epoch[ch])

        # pywt.cwt libera el GIL → threads reales en paralelo
        with ThreadPoolExecutor(max_workers=min(n_channels, 32)) as pool:
            list(pool.map(_cwt_ch, range(n_channels)))

        mean_per_freq = scalograms.mean(axis=(0, 2), keepdims=True)
        std_per_freq  = scalograms.std(axis=(0, 2), keepdims=True) + 1e-8
        scalograms = (scalograms - mean_per_freq) / std_per_freq

        return scalograms   

    def project_channels_pca(self, scalograms: np.ndarray) -> np.ndarray:
        """
        Proyecta 306 canales a 3 componentes usando PCA sobre la dimensión canal.
        Equivalente a la variante PCA-3 del paper ISBI (ligeramente peor que conv learned).

        Args:
            scalograms: (n_channels, n_freqs, T)

        Returns:
            projected: (3, n_freqs, T)
        """
        n_channels, n_freqs, T = scalograms.shape

        if not self.pca_fitted:
            # Ajustar PCA la primera vez (sobre datos de esta epoch como proxy)
            # En producción: ajustar sobre un conjunto representativo del train set
            data_2d = scalograms.reshape(n_channels, -1).T  # (n_freqs*T, n_channels)
            # SVD como PCA
            U, S, Vt = np.linalg.svd(data_2d - data_2d.mean(0), full_matrices=False)
            self.pca_components = Vt[:self.n_pca_components]  # (3, n_channels)
            self.pca_fitted = True

        # Proyección: (3, n_channels) @ (n_channels, n_freqs*T) → (3, n_freqs*T)
        data_flat = scalograms.reshape(n_channels, -1)  # (n_channels, n_freqs*T)
        projected = self.pca_components @ data_flat      # (3, n_freqs*T)
        projected = projected.reshape(3, n_freqs, T)     # (3, n_freqs, T)

        return projected.astype(np.float32)

    def resize_to_image(self, tensor_3d: np.ndarray) -> np.ndarray:
        """
        Redimensiona (3, n_freqs, T) a (3, 224, 224) mediante interpolación bilineal.

        Args:
            tensor_3d: (3, n_freqs, T)

        Returns:
            image: (3, img_size, img_size)
        """
        import torch.nn.functional as F_nn
        t = torch.from_numpy(tensor_3d).unsqueeze(0)  # (1, 3, n_freqs, T)
        resized = F_nn.interpolate(
            t,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).numpy()  # (3, img_size, img_size)

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        """
        Pipeline completo: epoch MEG → imagen RGB.

        Args:
            epoch: (n_channels, T) array float32 (ya preprocesado)

        Returns:
            image: (3, 224, 224) array float32 listo para modelos ImageNet
        """
        # Paso 1: CWT → escalogramas (computacionalmente costoso)
        scalograms = self.compute_all_scalograms(epoch)  # (306, 96, T)

        # Paso 2: Proyección de canales a 3 mapas espaciales
        if self.projection == "pca":
            # Variante estática PCA (más rápida, menor rendimiento)
            projected = self.project_channels_pca(scalograms)  # (3, 96, T)
        elif self.projection == "learned":
            # La proyección learnable (Conv 1×1) se aplica en el modelo nn.
            # Aquí simplemente tomamos 3 canales representativos como placeholder.
            # El módulo SensorMixer del modelo reemplazará esta operación.
            projected = scalograms[:3].copy()  # (3, 96, T) placeholder
        else:
            raise ValueError(f"Proyección desconocida: {self.projection}")

        # Paso 3: Resize bilineal a 224×224
        image = self.resize_to_image(projected)  # (3, 224, 224)

        # Paso 4: Normalización ImageNet (media y std de ImageNet)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

        # Re-escalar al rango [0, 1] antes de aplicar normalización ImageNet
        img_min = image.min(axis=(1, 2), keepdims=True)
        img_max = image.max(axis=(1, 2), keepdims=True)
        image = (image - img_min) / (img_max - img_min + 1e-8)

        image = (image - mean) / std

        return image.astype(np.float32)


###############################################################################
# Utilities de augmentación y carga de imágenes precomputadas
###############################################################################

# Parámetros de data augmentation de señal (se mantienen iguales al pipeline actual)
AUG_TEMPORAL_SHIFT_PROB = 0.5
AUG_TEMPORAL_SHIFT_FRAC = 0.10
AUG_AMPLITUDE_JITTER_PROB = 0.5
AUG_AMPLITUDE_JITTER_RANGE = 0.05
AUG_CHANNEL_DROPOUT_PROB = 0.3
AUG_CHANNEL_DROPOUT_FRAC = 0.10


def apply_signal_augmentation(
    epoch: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Aplica data augmentation al epoch MEG (antes de CWT), manteniendo los
    mismos hiperparámetros que ya se usan en entrenamiento.
    """
    epoch_aug = np.array(epoch, dtype=np.float32, copy=True)
    T = epoch_aug.shape[1]

    if rng is None:
        rand = np.random.rand
        randint = np.random.randint
        uniform = np.random.uniform
        choice = np.random.choice
    else:
        rand = rng.random
        randint = rng.integers
        uniform = rng.uniform
        choice = rng.choice

    # 1) Temporal shift (±10% de la ventana)
    if rand() < AUG_TEMPORAL_SHIFT_PROB:
        max_shift = max(1, int(AUG_TEMPORAL_SHIFT_FRAC * T))
        shift = int(randint(-max_shift, max_shift + 1))
        epoch_aug = np.roll(epoch_aug, shift, axis=1)

    # 2) Amplitude jitter (multiplicativo ±5%)
    if rand() < AUG_AMPLITUDE_JITTER_PROB:
        jitter = 1.0 + float(uniform(-AUG_AMPLITUDE_JITTER_RANGE, AUG_AMPLITUDE_JITTER_RANGE))
        epoch_aug = epoch_aug * jitter

    # 3) Channel dropout (zeroing del 10% de canales)
    if rand() < AUG_CHANNEL_DROPOUT_PROB:
        n_drop = int(AUG_CHANNEL_DROPOUT_FRAC * epoch_aug.shape[0])
        drop_idx = choice(epoch_aug.shape[0], n_drop, replace=False)
        epoch_aug[drop_idx] = 0.0

    return epoch_aug


# ==============================================================================
# SECCIÓN 4: DATASET PYTORCH CON GENERACIÓN ON-THE-FLY
# ==============================================================================

class MEGImageDataset(Dataset):
    """
    Dataset PyTorch que carga epochs MEG desde pnpl y genera imágenes TF on-the-fly.

    Flujo por sample:
      pnpl epoch (306, T)
        → MEGPreprocessor  (normalización)
        → MEGToImage       (CWT + proyección + resize)
        → tensor (3, 224, 224)

    La representación se genera siempre on-the-fly para no congelar una
    proyección fija antes de capas aprendibles posteriores.

    Data augmentation (siguiendo el paper ISBI 2026 y MEGConformer):
      - Temporal shift: ±10% de la longitud de la ventana
      - Frequency masking: máscara aleatoria en bandas de frecuencia
      - Amplitude jitter: ±5% de variación en amplitud
    """

    def __init__(
        self,
        pnpl_dataset,
        preprocessor: MEGPreprocessor,
        img_converter: MEGToImage,
        augment: bool = False,
        signal_projector: Optional[LinearSourceProjector] = None,
    ):
        self.pnpl_dataset  = pnpl_dataset
        self.preprocessor  = preprocessor
        self.img_converter = img_converter
        self.augment       = augment
        self.signal_projector = signal_projector

    def _process_sample(self, idx: int) -> Tuple[np.ndarray, int]:
        """Carga, preprocesa y convierte un sample a imagen."""
        sample = self.pnpl_dataset[idx]
        # pnpl devuelve (epoch, label) donde epoch shape = (n_channels, T)
        epoch, label = sample[0], sample[1]
        epoch = np.array(epoch, dtype=np.float32)
        label = reduce_label_to_scalar(label)

        # Preprocesado
        epoch = self.preprocessor(epoch)

        # Proyección sensor -> fuente/ROI antes de la CWT.
        if self.signal_projector is not None:
            epoch = self.signal_projector(epoch)

        # Augmentation (solo durante entrenamiento)
        if self.augment:
            epoch = self._apply_augmentation(epoch)

        # Conversión a imagen
        image = self.img_converter(epoch)  # (3, 224, 224)

        return image, label

    def _apply_augmentation(self, epoch: np.ndarray) -> np.ndarray:
        """Wrapper de compatibilidad hacia la función compartida de augmentación."""
        return apply_signal_augmentation(epoch)

    def __len__(self) -> int:
        return len(self.pnpl_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self._process_sample(idx)
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def build_dataloaders(
    train_pnpl, val_pnpl, test_pnpl,
    preprocessor: MEGPreprocessor,
    img_converter: MEGToImage,
    batch_size: int = 32,
    num_workers: int = 4,
    signal_projector: Optional[LinearSourceProjector] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construye los DataLoaders de entrenamiento, validación y test.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = MEGImageDataset(
        train_pnpl, preprocessor, img_converter,
        augment=True,
        signal_projector=signal_projector,
    )
    val_ds = MEGImageDataset(
        val_pnpl, preprocessor, img_converter,
        augment=False,
        signal_projector=signal_projector,
    )
    test_ds = MEGImageDataset(
        test_pnpl, preprocessor, img_converter,
        augment=False,
        signal_projector=signal_projector,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[INFO] Batches — Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def reduce_label_to_scalar(label) -> int:
    """
    Convierte labels de pnpl a un escalar por ventana.

    LibriBrainSpeech devuelve una etiqueta 0/1 por muestra temporal; este
    pipeline genera una sola imagen por ventana, así que entrenamos contra la
    etiqueta central de la ventana.
    """
    label_arr = np.asarray(label)
    if label_arr.ndim == 0 or label_arr.size == 1:
        return int(label_arr.reshape(-1)[0])

    center_idx = label_arr.size // 2
    return int(label_arr.reshape(-1)[center_idx])


# ==============================================================================
# SECCIÓN 5: DEFINICIÓN DE MODELOS CON PROYECCIÓN SENSOR-ESPACIO
# ==============================================================================

class SensorMixer(nn.Module):
    """
    Capa de mezcla de sensores: proyección 1×1 convolucional learnable.

    Implementa la operación central del paper ISBI 2026:
    "A learnable 1×1 convolutional projection across the sensor dimension"
    inspirada en channel-mixing de redes ConvMixer (Trockman & Kolter, 2022).

    Transforma: (batch, n_channels, n_freqs, T) → (batch, 3, n_freqs, T)

    A diferencia de PCA fija, esta proyección se aprende conjuntamente con
    el resto del modelo, capturando patrones distribuidos en el espacio de
    sensores que son informativos para la tarea específica.
    """

    def __init__(self, n_input_channels: int = 306, n_output_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=n_output_channels,
            kernel_size=1,
            bias=False,
        )
        # Inicialización: distribuir pesos uniformemente entre canales
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_freqs, T)
        Returns:
            (batch, 3, n_freqs, T)
        """
        return self.conv(x)


class MEGImageModel(nn.Module):
    """
    Modelo completo: SensorMixer + backbone preentrenado + cabeza de clasificación.

    La arquitectura tiene 3 componentes:
    1. SensorMixer: conv 1×1 learnable (306 → 3 canales)
    2. Backbone: modelo ImageNet preentrenado (ResNet, EfficientNet, ViT...)
    3. Classification head: capas densas para la tarea MEG

    El SensorMixer siempre se entrena desde cero (no hay preentrenamiento
    para proyectar 306 canales MEG a RGB). El backbone puede congelarse
    total o parcialmente según la estrategia elegida.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        n_classes: int = 39,
        n_meg_channels: int = 306,
        pretrained: bool = True,
        strategy: str = "partial_ft",   # "frozen" | "partial_ft" | "full_ft"
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.strategy      = strategy
        self.n_classes     = n_classes

        # ── Componente 1: SensorMixer (siempre se entrena) ────────────────────
        self.sensor_mixer = SensorMixer(
            n_input_channels=n_meg_channels,
            n_output_channels=3,
        )

        # ── Componente 2: Backbone preentrenado ───────────────────────────────
        self.backbone, feature_dim = self._build_backbone(backbone_name, pretrained)

        # ── Aplicar estrategia de congelación ─────────────────────────────────
        self._apply_strategy(strategy)

        # ── Componente 3: Cabeza de clasificación ─────────────────────────────
        output_dim = 1 if n_classes == 2 else n_classes
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim),
        )

    def _build_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Construye el backbone y devuelve (modelo, dimensión_features)."""
        weights = "pretrained" if pretrained else None

        if name == "resnet18":
            weights_obj = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights_obj)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()  # Eliminar cabeza original
            return model, feature_dim

        elif name == "efficientnet_b0":
            weights_obj = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b0(weights=weights_obj)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feature_dim

        elif name in ("vit_tiny", "vit_base") and TIMM_AVAILABLE:
            timm_name = {
                "vit_tiny": "vit_tiny_patch16_224",
                "vit_base": "vit_base_patch16_224",
            }[name]
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,  # Sin cabeza de clasificación
            )
            feature_dim = model.num_features
            return model, feature_dim

        else:
            raise ValueError(
                f"Backbone desconocido: {name}. "
                f"Opciones: resnet18, efficientnet_b0, vit_tiny, vit_base"
            )

    def _apply_strategy(self, strategy: str):
        """
        Aplica la estrategia de entrenamiento al backbone.

        Estrategias (basadas en ablación de Jhilal et al. 2026):
        ┌─────────────────┬──────────────────────────────────────────────────────────┐
        │ Estrategia      │ Qué se entrena                                           │
        ├─────────────────┼──────────────────────────────────────────────────────────┤
        │ frozen          │ Solo SensorMixer + classification head                   │
        │                 │ Backbone completamente congelado                         │
        │                 │ → Transfer learning puro, rápido pero menos flexible     │
        ├─────────────────┼──────────────────────────────────────────────────────────┤
        │ partial_ft      │ SensorMixer + último bloque backbone + classification    │
        │                 │ Capas tempranas/medias congeladas                        │
        │                 │ → MEJOR según ablación ISBI: balance rendimiento/overfitting│
        ├─────────────────┼──────────────────────────────────────────────────────────┤
        │ full_ft         │ Todo el modelo                                           │
        │                 │ → Mayor riesgo de overfitting, peor en LOSO              │
        └─────────────────┴──────────────────────────────────────────────────────────┘
        """
        if strategy == "frozen":
            # Congelar todo el backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif strategy == "partial_ft":
            # Congelar capas tempranas y medias, descongelar último bloque
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Descongelar último bloque (específico por arquitectura)
            if self.backbone_name == "resnet18":
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            elif self.backbone_name == "efficientnet_b0":
                # Descongelar último bloque de features
                for param in self.backbone.features[-2:].parameters():
                    param.requires_grad = True
            elif self.backbone_name in ("vit_tiny", "vit_base"):
                # Descongelar últimos 2 bloques transformer
                for block in self.backbone.blocks[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True

        elif strategy == "full_ft":
            # Todos los parámetros del backbone se actualizan
            for param in self.backbone.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")

    def _set_frozen_backbone_bn_eval(self):
        """Evita drift de running stats en BatchNorm congeladas."""
        for mod in self.backbone.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in mod.parameters()):
                    mod.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_frozen_backbone_bn_eval()
        return self

    def get_param_groups(self, lr_head: float = 1e-3, lr_backbone: float = 1e-4):
        """
        Devuelve grupos de parámetros con learning rates diferenciados.

        Siguiendo ISBI 2026:
          - LR alto (1e-3): SensorMixer y classification head (entrenados desde cero)
          - LR bajo (1e-4): capas descongeladas del backbone (ajuste fino suave)
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.sensor_mixer.parameters())
            + list(self.classifier.parameters())
        )

        param_groups = [
            {"params": head_params,     "lr": lr_head,    "name": "head"},
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]
        return param_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass completo.

        NOTA: Aquí x ya viene como imagen (3, 224, 224) desde el Dataset.
        El SensorMixer se aplica ANTES de la conversión a imagen en el Dataset
        (en el pipeline "learned"). En este forward, el SensorMixer
        actúa sobre la imagen directamente para poder ser aprendido end-to-end.

        Args:
            x: (batch, 3, 224, 224) imagen TF ya generada

        Returns:
            logits: (batch, n_classes)
        """
        # Backbone + pool global
        features = self.backbone(x)  # (batch, feature_dim)

        # Clasificación
        logits = self.classifier(features)  # (batch, n_classes)

        return logits


class MEGImageModelEndToEnd(nn.Module):
    """
    Variante end-to-end: el SensorMixer opera sobre los escalogramas raw
    (ANTES de crear la imagen), permitiendo aprendizaje conjunto.

    Flujo:
      (batch, 306, 96, T) → SensorMixer → (batch, 3, 96, T)
          → resize a (batch, 3, 224, 224) → backbone → classifier

    Esta variante es más precisa al paper ISBI 2026, donde la conv 1×1
    se aplica sobre los escalogramas (no sobre la imagen resizeada).
    Requiere pasar los escalogramas directamente al modelo.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        n_classes: int = 39,
        n_meg_channels: int = 306,
        n_freqs: int = 96,
        img_size: int = 224,
        pretrained: bool = True,
        strategy: str = "partial_ft",
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.img_size = img_size

        # SensorMixer sobre escalogramas (conv 1×1 en dim. canales)
        self.sensor_mixer = SensorMixer(n_meg_channels, 3)

        # Construir backbone (reutilizamos MEGImageModel internamente)
        _dummy = MEGImageModel(
            backbone_name, n_classes, 3, pretrained, strategy, dropout_rate
        )
        self.backbone   = _dummy.backbone
        self.classifier = _dummy.classifier
        self.backbone_name = backbone_name
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def get_param_groups(self, lr_head: float = 1e-3, lr_backbone: float = 1e-4):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.sensor_mixer.parameters())
            + list(self.classifier.parameters())
        )
        return [
            {"params": head_params,     "lr": lr_head,     "name": "head"},
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        ]

    def _set_frozen_backbone_bn_eval(self):
        for mod in self.backbone.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in mod.parameters()):
                    mod.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._set_frozen_backbone_bn_eval()
        return self

    def forward(self, scalograms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scalograms: (batch, n_channels, n_freqs, T) — escalogramas raw

        Returns:
            logits: (batch, n_classes)
        """
        # SensorMixer: 306 → 3 canales (conv 1×1 en dimensión de sensores)
        x = self.sensor_mixer(scalograms)  # (batch, 3, n_freqs, T)

        # Resize bilineal a 224×224
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)

        # Mantener la misma escala que en el pipeline CPU (MEGToImage):
        # min-max por canal y normalización ImageNet antes del backbone.
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        x = (x - self.imagenet_mean) / self.imagenet_std

        # Backbone ImageNet
        features = self.backbone(x)  # (batch, feature_dim)

        # Clasificación
        logits = self.classifier(features)

        return logits


# ==============================================================================
# SECCIÓN 6: ESTRATEGIAS DE TRANSFER LEARNING Y FINE-TUNING
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Configuración completa del experimento de entrenamiento.

    Basada en los mejores hiperparámetros reportados en:
      - Jhilal et al. (ISBI 2026): LR diferenciados, cosine annealing, 30 epochs
      - MEGConformer (NeurIPS 2025): AdamW, LR 1e-4, batch 256, early stopping
    """
    # ── Modelo ────────────────────────────────────────────────────────────────
    backbone: str = "resnet18"          # resnet18 | efficientnet_b0 | vit_tiny | vit_base
    pretrained: bool = True             # Usar pesos ImageNet
    strategy: str = "partial_ft"        # frozen | partial_ft | full_ft
    n_classes: int = 39                 # 2 (speech) | 39 (phoneme)
    dropout_rate: float = 0.5

    # ── Optimización ──────────────────────────────────────────────────────────
    lr_head: float = 1e-3               # LR para SensorMixer + classification head
    lr_backbone: float = 1e-4           # LR para capas descongeladas del backbone
    weight_decay: float = 5e-2          # L2 regularización (AdamW)
    n_epochs: int = 30
    batch_size: int = 32
    grad_clip: float = 1.0              # max-norm gradient clipping

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler: str = "cosine"           # cosine | step | none

    # ── Early stopping ────────────────────────────────────────────────────────
    patience: int = 10                  # epochs sin mejora antes de parar
    monitor: str = "f1_macro"           # métrica a monitorizar

    # ── Manejo de desbalanceo de clases ───────────────────────────────────────
    # Fonemas: muy desbalanceado (schwa /ə/ >> fonemas raros)
    # Estrategia ISNS: w_c ∝ 1/sqrt(n_c) (MEGConformer)
    use_class_weights: bool = True
    class_weight_method: str = "isns"   # "balanced" (sklearn) | "isns" (sqrt inversa)

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: str = "./results"
    experiment_name: str = "meg_tl_experiment"


def compute_class_weights_isns(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Calcula pesos de clase usando la regla ISNS (Inverse Square root of N Samples).
    w_c ∝ 1/sqrt(n_c), normalizado para que sum(w_c) = 1.

    Método del paper MEGConformer: superior a "balanced" para fonemas en LibriBrain.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)  # evitar división por cero
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * n_classes  # escalar
    return torch.tensor(weights, dtype=torch.float32)


def compute_binary_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Calcula pos_weight = n_neg / n_pos para BCEWithLogitsLoss."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    neg_count = max(counts[0], 1.0)
    pos_count = max(counts[1], 1.0)
    return torch.tensor([neg_count / pos_count], dtype=torch.float32)


class BCEWithLogitsLossWithSmoothing(nn.Module):
    """Binary cross-entropy con label smoothing determinista."""

    def __init__(
        self,
        smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer(
            "pos_weight",
            pos_weight.detach().clone() if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float().view_as(logits)
        if self.smoothing > 0:
            target = target * (1.0 - self.smoothing) + self.smoothing * 0.5

        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
        )


def build_criterion(
    n_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> nn.Module:
    if n_classes == 2:
        pos_weight = class_weights.to(DEVICE) if class_weights is not None else None
        return BCEWithLogitsLossWithSmoothing(
            smoothing=label_smoothing,
            pos_weight=pos_weight,
        )

    weights = class_weights.to(DEVICE) if class_weights is not None else None
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def logits_to_predictions(logits: torch.Tensor, n_classes: int) -> np.ndarray:
    if n_classes == 2 and logits.shape[1] == 1:
        return (torch.sigmoid(logits).squeeze(1) >= 0.5).long().cpu().numpy()

    return logits.argmax(dim=1).cpu().numpy()


def build_optimizer_and_scheduler(
    model: MEGImageModel,
    config: TrainingConfig,
    n_batches_per_epoch: int,
) -> Tuple:
    """
    Construye el optimizador AdamW con grupos de LR diferenciados y el scheduler.

    Grupos de learning rate (Jhilal et al. 2026):
      - LR alto (1e-3): SensorMixer + classification head
      - LR bajo (1e-4): capas descongeladas del backbone
    """
    param_groups = model.get_param_groups(
        lr_head=config.lr_head,
        lr_backbone=config.lr_backbone,
    )

    optimizer = AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.n_epochs,
            eta_min=1e-6,
        )
    else:
        scheduler = None

    return optimizer, scheduler


# ==============================================================================
# SECCIÓN 7: BUCLE DE ENTRENAMIENTO CON MÉTRICAS Y EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """Early stopping para evitar overfitting."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, value: float, model: nn.Module) -> bool:
        """
        Returns:
            True si debe pararse el entrenamiento, False si continuar.
        """
        improved = (
            value > self.best_value + self.min_delta if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter    = 0
            # Guardar estado del modelo
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Restaura el mejor estado del modelo guardado."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Entrena una epoch completa.

    Returns:
        dict con métricas: loss, accuracy, f1_macro
    """
    model.train()
    total_loss   = 0.0
    all_preds    = []
    all_labels   = []

    import torch.distributed as dist

    is_main = (not dist.is_initialized()) or dist.get_rank() == 0
    for batch_idx, (batch_x, batch_y) in enumerate(tqdm(loader, desc="Train", leave=False, disable=not is_main)):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        if batch_idx % 50 == 0:
            is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            if is_rank0:
                print(f"  [batch {batch_idx}/{len(loader)}]", flush=True)
    

        optimizer.zero_grad()

        logits = model(batch_x)
        loss   = criterion(logits, batch_y)

        loss.backward()

        # Gradient clipping (previene explosión de gradientes)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = logits_to_predictions(logits, n_classes)
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)

    return {"loss": avg_loss, "f1_macro": f1, "balanced_acc": bal_acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int,
) -> Dict[str, float]:
    """
    Evalúa el modelo en el conjunto de validación o test.

    Returns:
        dict con métricas: loss, accuracy, f1_macro, balanced_acc
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for batch_x, batch_y in tqdm(loader, desc="Eval", leave=False):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        loss   = criterion(logits, batch_y)

        total_loss += loss.item()
        preds = logits_to_predictions(logits, n_classes)
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)

    return {"loss": avg_loss, "f1_macro": f1, "balanced_acc": bal_acc}


def train_model(
    model: MEGImageModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, List]:
    """
    Bucle completo de entrenamiento con early stopping y logging.

    Returns:
        history: dict con listas de métricas por epoch (train y val)
    """
    model = model.to(DEVICE)

    # ── Función de pérdida ────────────────────────────────────────────────────
    # Speech usa BCEWithLogits con smoothing; phoneme mantiene CrossEntropy.
    criterion = build_criterion(config.n_classes, class_weights, label_smoothing=0.1)

    # ── Optimizador y scheduler ───────────────────────────────────────────────
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, len(train_loader)
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=config.patience,
        mode="max" if config.monitor in ("f1_macro", "balanced_acc") else "min",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_f1": [], "train_bal_acc": [],
        "val_loss":   [], "val_f1":   [], "val_bal_acc":   [],
    }

    print(f"\n{'='*60}")
    print(f"  Experimento: {config.experiment_name}")
    print(f"  Backbone: {config.backbone} | Estrategia: {config.strategy}")
    print(f"  Pretrained: {config.pretrained} | Clases: {config.n_classes}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.n_epochs + 1):

        # ── Entrenamiento ─────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            config.n_classes, config.grad_clip
        )

        # ── Validación ────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, criterion, DEVICE, config.n_classes)

        # ── Scheduler step ────────────────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1_macro"])
        history["train_bal_acc"].append(train_metrics["balanced_acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_bal_acc"].append(val_metrics["balanced_acc"])

        print(
            f"Epoch {epoch:03d}/{config.n_epochs} │ "
            f"Train Loss: {train_metrics['loss']:.4f} │ "
            f"Train F1: {train_metrics['f1_macro']:.4f} │ "
            f"Val Loss: {val_metrics['loss']:.4f} │ "
            f"Val F1: {val_metrics['f1_macro']:.4f} │ "
            f"Val Bal.Acc: {val_metrics['balanced_acc']:.4f}"
        )

        # ── Early stopping ────────────────────────────────────────────────────
        monitor_val = val_metrics[config.monitor]
        if early_stopper.step(monitor_val, model):
            print(f"\n[INFO] Early stopping activado en epoch {epoch}. "
                  f"Mejor {config.monitor}: {early_stopper.best_value:.4f}")
            break

    # Restaurar el mejor modelo
    early_stopper.restore_best(model)

    return history


# ==============================================================================
# SECCIÓN 8: EVALUACIÓN Y COMPARATIVA DE ESTRATEGIAS
# ==============================================================================

def run_experiment(
    backbone: str,
    strategy: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_classes: int,
    n_meg_channels: int = 306,
    pretrained: bool = True,
    n_epochs: int = 30,
    batch_size: int = 32,
    class_weights: Optional[torch.Tensor] = None,
    output_dir: str = "./results",
    representation: str = "sensor",
) -> Dict:
    """
    Ejecuta un experimento completo para una combinación backbone × estrategia.

    Returns:
        results: dict con métricas finales de test, historia de entrenamiento
                 y configuración del experimento
    """
    exp_name = f"{backbone}_{strategy}_{'pretrained' if pretrained else 'scratch'}"
    if representation != "sensor":
        exp_name = f"{representation}__{exp_name}"

    config = TrainingConfig(
        backbone=backbone,
        pretrained=pretrained,
        strategy=strategy,
        n_classes=n_classes,
        experiment_name=exp_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    # Construir modelo
    model = MEGImageModel(
        backbone_name=backbone,
        n_classes=n_classes,
        n_meg_channels=n_meg_channels,
        pretrained=pretrained,
        strategy=strategy,
        dropout_rate=config.dropout_rate,
    )

    # Contar parámetros entrenables
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parámetros: {trainable:,} entrenables / {total:,} totales "
          f"({100*trainable/total:.1f}%)")

    # Entrenamiento
    history = train_model(model, train_loader, val_loader, config, class_weights)

    # Evaluación final en test
    criterion = build_criterion(n_classes, class_weights, label_smoothing=0.0)
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, n_classes)

    print(f"\n[TEST] {exp_name}")
    print(f"  F1-macro:      {test_metrics['f1_macro']:.4f}")
    print(f"  Balanced Acc:  {test_metrics['balanced_acc']:.4f}")

    # Guardar checkpoint
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = f"{output_dir}/{exp_name}_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "test_metrics": test_metrics,
    }, checkpoint_path)
    print(f"  Checkpoint guardado: {checkpoint_path}")

    return {
        "experiment": exp_name,
        "backbone": backbone,
        "strategy": strategy,
        "pretrained": pretrained,
        "representation": representation,
        "n_meg_channels": n_meg_channels,
        "test_f1_macro": test_metrics["f1_macro"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "history": history,
        "model": model,
    }


def compare_strategies(results: List[Dict], output_dir: str = "./results"):
    """
    Genera una tabla comparativa de todos los experimentos y guarda las curvas
    de entrenamiento.

    Muestra la ablación equivalente a la Tabla 4 del paper ISBI 2026, adaptada
    a LibriBrain.
    """
    print("\n" + "="*70)
    print("  COMPARATIVA DE ESTRATEGIAS DE TRANSFER LEARNING")
    print("="*70)
    print(f"{'Experimento':<50} {'F1-macro':>10} {'Bal.Acc':>10}")
    print("-"*70)

    # Ordenar por F1-macro descendente
    results_sorted = sorted(results, key=lambda r: r["test_f1_macro"], reverse=True)

    for r in results_sorted:
        pretrained_str = "ImageNet" if r["pretrained"] else "Random"
        repr_str = r.get("representation", "sensor")
        exp_label = f"{repr_str}: {r['backbone']} + {r['strategy']} ({pretrained_str})"
        print(f"{exp_label:<50} {r['test_f1_macro']:>10.4f} {r['test_balanced_acc']:>10.4f}")

    print("="*70)

    # Gráfica de curvas de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for r, color in zip(results, colors):
        label = f"{r.get('representation', 'sensor')}: {r['backbone']} + {r['strategy']}"
        axes[0].plot(r["history"]["val_f1"],    label=label, color=color)
        axes[1].plot(r["history"]["val_loss"],  label=label, color=color)

    axes[0].set_title("Val F1-macro por epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("F1-macro")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Val Loss por epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/training_curves_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Curvas de entrenamiento guardadas: {plot_path}")

    return results_sorted


def run_best_source_retrain(
    best_result: Dict,
    train_pnpl,
    val_pnpl,
    test_pnpl,
    preprocessor: MEGPreprocessor,
    n_classes: int,
    sensor_n_channels: int,
    class_weights: Optional[torch.Tensor],
    args,
) -> Dict:
    """
    Repite el entrenamiento ganador usando una proyección lineal antes de la CWT.

    Esta fase está pensada para filtros LCMV/DSS ya calculados fuera del script.
    En MNE-Python, lo más práctico es exportar la matriz final ROI/source x sensor
    y pasarla con --source_projection_path.
    """
    source_projector = LinearSourceProjector(
        args.source_projection_path,
        name=args.source_variant_name,
    )
    source_projector.validate_input_channels(sensor_n_channels)

    print("\n[PASO 9] Reentrenando el mejor modelo con proyección fuente antes de CWT...")
    print(f"  - Proyección: {args.source_projection_path}")
    print(
        f"  - Canales: {source_projector.n_inputs} sensores -> "
        f"{source_projector.n_outputs} fuentes/ROIs"
    )
    print(
        f"  - Config ganadora: {best_result['backbone']} | "
        f"{best_result['strategy']} | "
        f"{'ImageNet' if best_result['pretrained'] else 'RandomInit'}"
    )

    source_img_converter = MEGToImage(
        sfreq=250.0,
        n_freqs=args.n_freqs,
        f_min=1.0,
        f_max=125.0,
        img_size=224,
        wavelet="cmor1.5-1.0",
        projection="pca",
    )

    source_train_loader, source_val_loader, source_test_loader = build_dataloaders(
        train_pnpl,
        val_pnpl,
        test_pnpl,
        preprocessor,
        source_img_converter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        signal_projector=source_projector,
    )

    return run_experiment(
        backbone=best_result["backbone"],
        strategy=best_result["strategy"],
        train_loader=source_train_loader,
        val_loader=source_val_loader,
        test_loader=source_test_loader,
        n_classes=n_classes,
        n_meg_channels=source_projector.n_outputs,
        pretrained=best_result["pretrained"],
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        class_weights=class_weights,
        output_dir=args.output_dir,
        representation=args.source_variant_name,
    )


def _load_sensor_xyz(sensor_xyz_path: str, n_channels: int) -> np.ndarray:
    """Carga coordenadas xyz de sensores y valida que coincidan con el modelo."""
    with open(sensor_xyz_path, "r", encoding="utf-8") as f:
        coords = np.asarray(json.load(f), dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"sensor_xyz debe tener shape (n_channels, 3); recibido {coords.shape}"
        )
    if coords.shape[0] != n_channels:
        raise ValueError(
            f"sensor_xyz tiene {coords.shape[0]} sensores, pero SensorMixer usa "
            f"{n_channels} canales"
        )
    return coords


def _aggregate_sensor_weights(
    weights: np.ndarray,
    coords: np.ndarray,
    decimals: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Agrupa canales MEG con la misma coordenada física.

    LibriBrain usa 306 canales y muchas coordenadas se repiten en tripletes
    para representar componentes/orientaciones del mismo sensor. Para interpretar
    regiones, la importancia física se calcula como norma L2 de todos los pesos
    que llegan a ese punto.
    """
    rounded = np.round(coords, decimals=decimals)
    _, first_idx, inverse = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_idx)

    grouped_coords = []
    grouped_rgb = []
    grouped_importance = []
    grouped_indices = []

    for group_id in order:
        idx = np.flatnonzero(inverse == group_id)
        grouped_indices.append(idx)
        grouped_coords.append(coords[idx].mean(axis=0))
        grouped_rgb.append(weights[:, idx].mean(axis=1))
        grouped_importance.append(float(np.linalg.norm(weights[:, idx])))

    return (
        np.asarray(grouped_coords, dtype=np.float32),
        np.asarray(grouped_rgb, dtype=np.float32),
        np.asarray(grouped_importance, dtype=np.float32),
        grouped_indices,
    )


def _scatter_sensor_map(
    ax,
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    symmetric: bool = False,
):
    """Dibuja un mapa 2D de sensores con escalado estable de color."""
    if symmetric:
        vmax = float(np.nanmax(np.abs(values)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    sizes = 35.0 + 180.0 * (
        (np.abs(values) - np.nanmin(np.abs(values)))
        / (np.ptp(np.abs(values)) + 1e-8)
    )
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=sizes,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    return sc


def visualize_sensor_mixer_weights(
    model: nn.Module,
    sensor_xyz_path: str,
    output_dir: str,
    experiment_name: str,
    top_k: int = 20,
) -> Dict[str, str]:
    """
    Proyecta los pesos aprendidos de SensorMixer al espacio físico de sensores.

    Guarda:
      - sensor_mixer_importance.png: importancia L2 agregada por sensor físico.
      - sensor_mixer_rgb_weights.png: mapas firmados para las 3 salidas RGB.
      - sensor_mixer_top_sensors.csv: ranking de sensores más informativos.
    """
    if not hasattr(model, "sensor_mixer") or not hasattr(model.sensor_mixer, "conv"):
        raise ValueError("El modelo no contiene un módulo sensor_mixer.conv")

    weight_tensor = model.sensor_mixer.conv.weight.detach().cpu()
    weights = weight_tensor.squeeze(-1).squeeze(-1).numpy()  # (3, n_channels)
    if weights.ndim != 2:
        raise ValueError(f"Pesos SensorMixer inesperados: shape {weights.shape}")

    n_output_channels, n_channels = weights.shape
    coords = _load_sensor_xyz(sensor_xyz_path, n_channels)
    grouped_coords, grouped_rgb, importance, grouped_indices = _aggregate_sensor_weights(
        weights, coords
    )

    out_dir = Path(output_dir) / "sensor_mixer"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_exp_name = experiment_name.replace("/", "_").replace(" ", "_")

    csv_path = out_dir / f"{safe_exp_name}_top_sensors.csv"
    rank = np.argsort(importance)[::-1]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "sensor_group", "channel_indices", "x", "y", "z",
            "importance_l2", "weight_out0", "weight_out1", "weight_out2",
        ])
        for row_rank, sensor_idx in enumerate(rank[:top_k], start=1):
            rgb = grouped_rgb[sensor_idx]
            padded_rgb = np.pad(rgb, (0, max(0, 3 - len(rgb))))[:3]
            writer.writerow([
                row_rank,
                int(sensor_idx),
                " ".join(map(str, grouped_indices[sensor_idx].tolist())),
                float(grouped_coords[sensor_idx, 0]),
                float(grouped_coords[sensor_idx, 1]),
                float(grouped_coords[sensor_idx, 2]),
                float(importance[sensor_idx]),
                float(padded_rgb[0]),
                float(padded_rgb[1]),
                float(padded_rgb[2]),
            ])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    views = [
        ((0, 1), "Vista axial", "x izquierda-derecha", "y anterior-posterior"),
        ((0, 2), "Vista coronal", "x izquierda-derecha", "z inferior-superior"),
        ((1, 2), "Vista sagital", "y anterior-posterior", "z inferior-superior"),
    ]
    for ax, (dims, title, xlabel, ylabel) in zip(axes, views):
        sc = _scatter_sensor_map(
            ax,
            grouped_coords[:, dims],
            importance,
            title,
            xlabel,
            ylabel,
            cmap="viridis",
            symmetric=False,
        )
    fig.suptitle(f"SensorMixer: importancia L2 por sensor - {experiment_name}")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, label="||pesos||2")
    importance_path = out_dir / f"{safe_exp_name}_importance.png"
    fig.savefig(importance_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, n_output_channels, figsize=(5.2 * n_output_channels, 4.8))
    if n_output_channels == 1:
        axes = [axes]
    for out_idx, ax in enumerate(axes):
        sc = _scatter_sensor_map(
            ax,
            grouped_coords[:, [0, 1]],
            grouped_rgb[:, out_idx],
            f"Salida {out_idx}",
            "x izquierda-derecha",
            "y anterior-posterior",
            cmap="coolwarm",
            symmetric=True,
        )
    fig.suptitle(f"SensorMixer: pesos firmados por salida - {experiment_name}")
    fig.colorbar(sc, ax=np.asarray(axes).ravel().tolist(), shrink=0.85, label="peso medio")
    rgb_path = out_dir / f"{safe_exp_name}_rgb_weights.png"
    fig.savefig(rgb_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "importance_png": str(importance_path),
        "rgb_weights_png": str(rgb_path),
        "top_sensors_csv": str(csv_path),
    }


# ==============================================================================
# SECCIÓN 9: SCRIPT PRINCIPAL (MAIN)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer Learning desde ImageNet para MEG LibriBrain"
    )
    parser.add_argument("--task",     type=str, default="phoneme",
                        choices=["speech", "phoneme"],
                        help="Tarea de decodificación")
    parser.add_argument("--model",    type=str, default="resnet18",
                        choices=["resnet18", "efficientnet_b0", "vit_tiny", "vit_base", "all"],
                        help="Backbone a usar ('all' para comparar todos)")
    parser.add_argument("--strategy", type=str, default="partial_ft",
                        choices=["frozen", "partial_ft", "full_ft", "all"],
                        help="Estrategia de fine-tuning ('all' para comparar todas)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Usar pesos ImageNet preentrenados")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Entrenar desde cero (sin ImageNet)")
    parser.add_argument("--data_path",  type=str, default="./libribrain_data")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--n_epochs",   type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Workers de DataLoader")
    parser.add_argument("--n_freqs",    type=int, default=96,
                        help="Bins de frecuencia para CWT (96 como en ISBI 2026)")
    parser.add_argument("--source_retrain_best", action="store_true",
                        help="Al final, repetir el mejor entrenamiento con proyección fuente antes de CWT")
    parser.add_argument("--source_projection_path", type=str, default=None,
                        help="Matriz W sensor->fuente/ROI (.npy/.npz/.pt/.json) para la segunda pasada")
    parser.add_argument("--source_variant_name", type=str, default="source_lcmv",
                        help="Prefijo/nombre de la variante fuente en resultados")
    parser.add_argument("--sensor_xyz_path", type=str, default="./libribrain/sensor_xyz.json",
                        help="Coordenadas xyz de sensores para visualizar SensorMixer")
    parser.add_argument("--top_sensor_k", type=int, default=20,
                        help="Número de sensores físicos a guardar en el ranking CSV")
    parser.add_argument("--no_sensor_mixer_viz", action="store_true",
                        help="No generar mapas de pesos de SensorMixer al final")
    return parser.parse_args()

def get_labels_fast(dataset) -> np.ndarray:
    """Extrae labels sin leer señales MEG."""
    if hasattr(dataset, 'phoneme_to_id'):
        return np.array([
            dataset.phoneme_to_id[s[-1].rsplit('_', 1)[0]]
            for s in dataset.samples
        ])

    if hasattr(dataset, 'samples'):
        labels = np.array([reduce_label_to_scalar(s[-1]) for s in dataset.samples])
        if np.isin(labels, [0, 1]).all():
            return labels

    raise NotImplementedError(
        f"get_labels_fast no implementado para {type(dataset).__name__}. "
        "Inspeccionar dataset.samples[0] y añadir el caso correspondiente."
    )

def main():
    args = parse_args()
    run_source_retrain = args.source_retrain_best or args.source_projection_path is not None
    if run_source_retrain and args.source_projection_path is None:
        raise ValueError(
            "--source_retrain_best requiere --source_projection_path con una matriz "
            "W sensor->fuente/ROI."
        )

    print("\n" + "="*60)
    print("  MEG TRANSFER LEARNING — LibriBrain Dataset")
    print("="*60)

    # ── PASO 1: Cargar datos con pnpl ─────────────────────────────────────────
    print("\n[PASO 1] Cargando datos LibriBrain...")

    train_cfg = LibriBrainConfig(args.data_path, args.task, "train")
    val_cfg   = LibriBrainConfig(args.data_path, args.task, "validation")
    test_cfg  = LibriBrainConfig(args.data_path, args.task, "test")

    train_pnpl, n_classes, n_channels = load_libribrain(train_cfg)
    val_pnpl,   _,         _          = load_libribrain(val_cfg)
    test_pnpl,  _,         _          = load_libribrain(test_cfg)

    # ── PASO 2: Configurar preprocesado ─────────────────────────────────────
    print("\n[PASO 2] Configurando preprocesado MEG...")

    # Instance normalization: clave para generalización en holdout (MEGConformer)
    preprocessor = MEGPreprocessor(
        use_instance_norm=True,
        baseline_samples=None,  # LibriBrain no tiene pre-cue definido
        clip_std=5.0,
    )

    # ── PASO 3: Configurar conversión a imagen TF ───────────────────────────
    print("\n[PASO 3] Configurando representación imagen tiempo-frecuencia (CWT)...")
    print(f"  - CWT Morlet: {args.n_freqs} frecuencias log, 1–125 Hz")
    print(f"  - Imagen destino: 224×224×3 (compatible con ImageNet)")
    print("  - Generación on-the-fly en cada batch")

    img_converter = MEGToImage(
        sfreq=250.0,
        n_freqs=args.n_freqs,
        f_min=1.0,
        f_max=125.0,     # Nyquist para fs=250 Hz
        img_size=224,
        wavelet="cmor1.5-1.0",  # Morlet complejo (ISBI 2026)
        projection="pca",
    )

    # ── PASO 4: Construir DataLoaders ─────────────────────────────────────────
    print("\n[PASO 4] Construyendo DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_pnpl, val_pnpl, test_pnpl,
        preprocessor, img_converter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── PASO 5: Pesos de clase para loss ponderada ────────────────────────────
    print("\n[PASO 5] Calculando pesos de clase...")
    # Extraer etiquetas del conjunto de entrenamiento para calcular pesos
    train_labels = get_labels_fast(train_pnpl)
    if n_classes == 2:
        class_weights = compute_binary_pos_weight(train_labels)
        print(f"  pos_weight BCE speech: {class_weights.item():.4f}")
    else:
        class_weights = compute_class_weights_isns(train_labels, n_classes)
        print(f"  Pesos ISNS calculados para {n_classes} clases")

    # ── PASO 6: Definir experimentos ──────────────────────────────────────────
    print("\n[PASO 6] Configurando experimentos...")

    if args.model == "all":
        backbones = ["resnet18", "efficientnet_b0"]
        if TIMM_AVAILABLE:
            backbones.append("vit_tiny")
    else:
        backbones = [args.model]

    if args.strategy == "all":
        strategies = ["frozen", "partial_ft", "full_ft"]
    else:
        strategies = [args.strategy]

    # Ablación: también probar sin preentrenamiento (como en ISBI 2026)
    pretrained_options = [True]
    if not args.pretrained:
        pretrained_options = [False]
    elif args.strategy == "all":
        pretrained_options = [True, False]  # Comparar con y sin ImageNet

    experiments = [
        (b, s, p)
        for b in backbones
        for s in strategies
        for p in pretrained_options
    ]
    print(f"  Total de experimentos: {len(experiments)}")
    for b, s, p in experiments:
        print(f"    • {b} + {s} + {'ImageNet' if p else 'RandomInit'}")

    # ── PASO 7: Ejecutar experimentos ─────────────────────────────────────────
    print("\n[PASO 7] Ejecutando experimentos de transfer learning...")

    all_results = []
    for backbone, strategy, pretrained in experiments:
        print(f"\n{'─'*60}")
        print(f"  Experimento: {backbone} | {strategy} | "
              f"{'ImageNet' if pretrained else 'RandomInit'}")
        print(f"{'─'*60}")

        result = run_experiment(
            backbone=backbone,
            strategy=strategy,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            n_classes=n_classes,
            n_meg_channels=n_channels,
            pretrained=pretrained,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            class_weights=class_weights,
            output_dir=args.output_dir,
        )
        all_results.append(result)

    # ── PASO 8: Comparativa y visualización ───────────────────────────────────
    print("\n[PASO 8] Comparando resultados...")
    ranked = compare_strategies(all_results, output_dir=args.output_dir)

    # Mejor resultado
    best = ranked[0]
    print(f"\n{'='*60}")
    print(f"  MEJOR RESULTADO:")
    print(f"  Experimento: {best['experiment']}")
    print(f"  F1-macro:    {best['test_f1_macro']:.4f}")
    print(f"  Bal. Acc:    {best['test_balanced_acc']:.4f}")
    print(f"{'='*60}\n")

    source_result = None
    if run_source_retrain:
        source_result = run_best_source_retrain(
            best_result=best,
            train_pnpl=train_pnpl,
            val_pnpl=val_pnpl,
            test_pnpl=test_pnpl,
            preprocessor=preprocessor,
            n_classes=n_classes,
            sensor_n_channels=n_channels,
            class_weights=class_weights,
            args=args,
        )
        all_results.append(source_result)
        print("\n[COMPARACIÓN SENSOR VS FUENTE]")
        print(
            f"  Sensor best: {best['test_f1_macro']:.4f} F1 | "
            f"{best['test_balanced_acc']:.4f} Bal.Acc"
        )
        print(
            f"  {args.source_variant_name}: {source_result['test_f1_macro']:.4f} F1 | "
            f"{source_result['test_balanced_acc']:.4f} Bal.Acc"
        )

    if not args.no_sensor_mixer_viz:
        print("[PASO 10] Visualizando pesos de SensorMixer...")
        try:
            viz_paths = visualize_sensor_mixer_weights(
                model=best["model"],
                sensor_xyz_path=args.sensor_xyz_path,
                output_dir=args.output_dir,
                experiment_name=best["experiment"],
                top_k=args.top_sensor_k,
            )
            print("  Mapas SensorMixer guardados:")
            for label, path in viz_paths.items():
                print(f"    - {label}: {path}")
            print(
                "  Nota: en la ruta PCA actual, MEGImageModel.forward() no usa "
                "SensorMixer; para interpretar regiones, entrena/evalúa la variante "
                "end-to-end donde la conv 1×1 recibe escalogramas raw."
            )
        except Exception as exc:
            warnings.warn(f"No se pudo visualizar SensorMixer: {exc}")


if __name__ == "__main__":
    main()


# ==============================================================================
# APÉNDICE: CÓMO AMPLIAR ESTE SCRIPT
# ==============================================================================
"""
EXTENSIONES RECOMENDADAS (líneas futuras del paper ISBI 2026):

1. DOMAIN ADAPTATION
   ─────────────────
   Usar TTA (Test-Time Augmentation) con promedios de múltiples augmentaciones
   de la misma epoch para reducir varianza en inferencia.

   Ejemplo (TTA con 10 augmentaciones):
       preds = []
       for _ in range(10):
           aug_img = augment_epoch(epoch)
           preds.append(model(aug_img).softmax(-1))
       final_pred = torch.stack(preds).mean(0).argmax(-1)

2. SELF-SUPERVISED PRETRAINING SOBRE MEG
   ───────────────────────────────────────
   Preentrenar un encoder sobre señales MEG sin etiquetas usando:
     - Masked Autoencoder (MAE): reconstruir patches enmascarados del escalograma
     - Contrastive Learning: contrastar augmentaciones del mismo epoch
   
   Luego hacer fine-tuning supervisado sobre las tareas de LibriBrain.
   
   Referencia: MEG-XL (Jayalath & Parker Jones, 2026) - pretraining con
   150s de contexto MEG mejora significativamente word decoding.

3. ENSEMBLE DE SEMILLAS (MEGConformer)
   ──────────────────────────────────────
   Entrenar 5-10 modelos con distintas semillas y promediar sus predicciones:
   
       all_logits = [model_k(x) for model_k in ensemble_models]
       final_pred = torch.stack(all_logits).mean(0).argmax(-1)
   
   Mejora reportada: +15% en F1-macro en el holdout de LibriBrain.

4. TOPOGRAFÍAS DE PESOS DE SENSORES
   ───────────────────────────────────
   Visualizar la conv 1×1 aprendida (SensorMixer) proyectando sus pesos
   de vuelta al espacio de sensores para interpretar qué regiones cerebrales
   son más informativas.

5. FUENTE EN LUGAR DE SENSOR SPACE
   ────────────────────────────────
   Proyectar señales al espacio fuente antes de la CWT usando MNE-Python
   (beamforming LCMV o DSS). El paper ISBI reporta rendimiento equivalente,
   pero la interpretabilidad mejora.
"""
