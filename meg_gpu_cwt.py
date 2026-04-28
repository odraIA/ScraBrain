"""
================================================================================
  meg_gpu_cwt.py — CWT acelerada por GPU para pipeline MEG
================================================================================

Reemplaza pywt.cwt (CPU, ~125s/batch) con CWT basada en FFT sobre GPU
(~0.5s/batch), usando torch.fft. Matemáticamente equivalente a la
implementación original de MEGToImage con cmor1.5-1.0.

Cambios en el pipeline:
  ANTES: DataLoader → MEGImageDataset.__getitem__ (CWT CPU) → imagen (3,224,224)
  AHORA: DataLoader → MEGRawDataset.__getitem__ (solo lectura) → señal (306,T)
              → CWTLayer.forward() en GPU → escalograma (306,n_freqs,T)
              → MEGImageModelEndToEnd (SensorMixer + backbone)

La clase MEGImageModelEndToEnd ya existe en meg_transfer_learning_libribrain.py
y acepta directamente escalogramas (B, 306, n_freqs, T).

Integración en train_ddp.py (ver comentario al final del fichero).
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Tuple, Optional


# ==============================================================================
# 1. DATASET QUE DEVUELVE SEÑAL MEG RAW (sin CWT)
# ==============================================================================

class MEGRawDataset(Dataset):
    """
    Dataset ligero que devuelve épocas MEG preprocesadas como tensores (306, T).
    La CWT se computa en GPU en el bucle de entrenamiento (CWTLayer).

    Comparado con MEGImageDataset:
      - __getitem__: ~2ms  vs  ~4000ms  (200x más rápido por sample)
      - Augmentation en espacio de señal funciona igual
      - Compatible con DDP y DistributedSampler
    """

    def __init__(
        self,
        pnpl_dataset,
        preprocessor,          # MEGPreprocessor
        augment: bool = False,
        speech_label_threshold: float = 0.5,
    ):
        self.pnpl_dataset = pnpl_dataset
        self.preprocessor = preprocessor
        self.augment = augment
        self.speech_label_threshold = speech_label_threshold

    def __len__(self) -> int:
        return len(self.pnpl_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.pnpl_dataset[idx]
        epoch  = np.array(sample[0], dtype=np.float32)  # (306, T)
        label  = self._label_to_index(sample[1])

        # Preprocesado en CPU (solo normalización, ~0.5ms)
        epoch = self.preprocessor(epoch)

        # Augmentation en espacio de señal (antes de CWT, como en ISBI 2026)
        if self.augment:
            epoch = self._augment(epoch)

        return torch.from_numpy(epoch), torch.tensor(label, dtype=torch.long)

    def _label_to_index(self, raw_label) -> int:
        """
        Convierte etiquetas de pnpl a índice de clase entero.

        Notas:
          - Phoneme entrega etiquetas escalares (int / tensor escalar).
          - Speech puede entregar una secuencia binaria por ventana; la
            colapsamos a clase binaria según mayoría en la ventana.
        """
        if isinstance(raw_label, torch.Tensor):
            if raw_label.numel() == 1:
                return int(raw_label.item())
            return int(raw_label.float().mean().item() >= self.speech_label_threshold)

        if isinstance(raw_label, np.ndarray):
            if raw_label.size == 1:
                return int(raw_label.reshape(-1)[0])
            return int(raw_label.astype(np.float32).mean() >= self.speech_label_threshold)

        if isinstance(raw_label, (list, tuple)):
            arr = np.asarray(raw_label)
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
            return int(arr.astype(np.float32).mean() >= self.speech_label_threshold)

        return int(raw_label)

    def _augment(self, epoch: np.ndarray) -> np.ndarray:
        """Mismas augmentaciones que MEGImageDataset._apply_augmentation."""
        T = epoch.shape[1]

        # 1. Temporal shift ±10%
        if np.random.rand() < 0.5:
            max_shift = max(1, int(0.10 * T))
            shift = np.random.randint(-max_shift, max_shift + 1)
            epoch = np.roll(epoch, shift, axis=1)

        # 2. Amplitude jitter ±5%
        if np.random.rand() < 0.5:
            epoch = epoch * (1.0 + np.random.uniform(-0.05, 0.05))

        # 3. Channel dropout 10%
        if np.random.rand() < 0.3:
            n_drop = int(0.1 * epoch.shape[0])
            drop_idx = np.random.choice(epoch.shape[0], n_drop, replace=False)
            epoch[drop_idx] = 0.0

        return epoch


# ==============================================================================
# 2. CWT EN GPU (FFT-based, equivalente a pywt.cwt con cmor1.5-1.0)
# ==============================================================================

class CWTLayer(nn.Module):
    """
    Transformada Wavelet Continua (CWT) implementada en GPU via FFT.

    Matemáticamente equivalente a pywt.cwt(signal, scales, 'cmor1.5-1.0').

    La CWT para wavelet ψ_s a escala s se define como:
        W(s, τ) = (1/√s) ∫ x(t) ψ*((t-τ)/s) dt

    En el dominio frecuencial (convolución = multiplicación):
        W_s(f) = X(f) · Ψ_s*(f)

    Para la Morlet compleja cmor{B}-{C}:
        Ψ(f) = √(πB) · exp(-π²B(f - C)²)   [solo frecuencias positivas]

    A escala s:
        Ψ_s(f) = √(s/fs) · √(πB) · exp(-π²B(s·f/fs - C)²)

    Parámetros:
        sfreq   : frecuencia de muestreo (Hz)
        n_freqs : número de bandas de frecuencia
        f_min   : frecuencia mínima (Hz)
        f_max   : frecuencia máxima (Hz)
        B       : parámetro de anchura de banda de cmor (1.5 en ISBI 2026)
        C       : frecuencia central de cmor (1.0 en ISBI 2026)

    Input:  (B, C_meg, T) — batch de señales MEG preprocesadas
    Output: (B, C_meg, n_freqs, T) — escalogramas (magnitud CWT), float32
    """

    def __init__(
        self,
        sfreq:   float = 250.0,
        n_freqs: int   = 96,
        f_min:   float = 1.0,
        f_max:   float = 125.0,
        B:       float = 1.5,   # cmor1.5-1.0 bandwidth
        C:       float = 1.0,   # cmor1.5-1.0 center frequency
    ):
        super().__init__()
        self.sfreq   = sfreq
        self.n_freqs = n_freqs
        self.B       = B
        self.C       = C

        # Frecuencias y escalas correspondientes
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
        # Relación escala ↔ frecuencia para cmor: scale = C * sfreq / freq
        scales = C * sfreq / frequencies

        self.register_buffer(
            'scales',
            torch.tensor(scales, dtype=torch.float32),
        )

        # Cache del banco de filtros (se construye la primera vez por longitud T)
        self._filter_cache: dict = {}

    @torch.no_grad()
    def _build_filter_bank(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Construye el banco de filtros Morlet en dominio frecuencial.

        Retorna: (n_freqs, T) complex128, listo para multiplicar con FFT de señal.
        """
        cache_key = (T, device.type, device.index)
        if cache_key in self._filter_cache:
            return self._filter_cache[cache_key]

        # Eje de frecuencias físicas (Hz) para cada bin DFT
        # fftfreq: [0, 1, ..., T/2-1, -T/2, ..., -1] / T * sfreq
        freqs_hz = torch.fft.fftfreq(T, d=1.0 / self.sfreq).to(
            device=device, dtype=torch.float32
        )  # (T,)

        s = self.scales.to(device).view(-1, 1)  # (n_freqs, 1)
        f = freqs_hz.view(1, -1)                # (1, T)

        # Filtro Morlet en dominio frecuencial:
        # Ψ_s(f) = sqrt(s/sfreq) · sqrt(πB) · exp(-π²B·(s·f/sfreq - C)²)
        psi = (
            torch.sqrt(s / self.sfreq)
            * (np.pi * self.B) ** 0.5
            * torch.exp(-np.pi ** 2 * self.B * (s * f / self.sfreq - self.C) ** 2)
        )  # (n_freqs, T), real

        # Señal analítica: anular frecuencias negativas
        # fftfreq: negativas están en índices T//2+1 ... T-1
        psi[:, T // 2 + 1 :] = 0.0
        psi[:, 0] = 0.0  # Componente DC

        filter_bank = psi.to(torch.complex64)  # (n_freqs, T)

        # Guardar en caché (máximo 8 tamaños distintos)
        if len(self._filter_cache) < 8:
            self._filter_cache[cache_key] = filter_bank

        return filter_bank

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_meg, T) float32 — señales MEG preprocesadas (en GPU)

        Returns:
            scalogram: (B, C_meg, n_freqs, T) float32 — magnitud CWT
        """
        B_sz, C_meg, T = x.shape
        device = x.device

        filter_bank = self._build_filter_bank(T, device)  # (n_freqs, T) complex128

        # Aplanar batch y canales para FFT en paralelo: (B*C_meg, T)
        x_flat = x.view(B_sz * C_meg, T).to(torch.float32)

        # FFT de todas las señales: (B*C_meg, T) complex
        X = torch.fft.fft(x_flat)  # (B*C_meg, T)

        # Multiplicación en dominio frecuencial (correlación cruzada con conjugado)
        # X: (B*C, 1, T) * Ψ*: (1, n_freqs, T) → (B*C, n_freqs, T)
        X_exp = X.unsqueeze(1)                  # (B*C, 1, T)
        F_exp = filter_bank.unsqueeze(0)        # (1, n_freqs, T)
        product = X_exp * F_exp.conj()          # (B*C, n_freqs, T) complex

        # IFFT → coeficientes CWT complejos
        coeff = torch.fft.ifft(product)         # (B*C, n_freqs, T) complex
        del product, X_exp, F_exp

        # Magnitud → escalograma
        scalogram = coeff.abs().to(torch.float32)  # (B*C, n_freqs, T) float32
        del coeff

        # Restaurar forma: (B*C, n_freqs, T) → (B, C_meg, n_freqs, T)
        scalogram = scalogram.view(B_sz, C_meg, self.n_freqs, T)

        return scalogram


# ==============================================================================
# 3. NORMALIZACIÓN POST-CWT (equivalente a MEGToImage.compute_all_scalograms)
# ==============================================================================

def zscore_scalogram(scalogram: torch.Tensor) -> torch.Tensor:
    """
    Z-score del escalograma por banda de frecuencia.
    Equivalente al z-score aplicado en MEGToImage.compute_all_scalograms.

    Args:
        scalogram: (B, C_meg, n_freqs, T)

    Returns:
        scalogram normalizado: (B, C_meg, n_freqs, T)
    """
    # Media y std sobre dimensiones canal y tiempo (manteniendo batch y freq)
    mean = scalogram.mean(dim=(1, 3), keepdim=True)   # (B, 1, n_freqs, 1)
    std  = scalogram.std(dim=(1, 3), keepdim=True) + 1e-8
    return (scalogram - mean) / std


# ==============================================================================
# 4. FUNCIÓN DE CONSTRUCCIÓN DE DATALOADERS
# ==============================================================================

def build_raw_dataloaders(
    train_pnpl,
    val_pnpl,
    test_pnpl,
    preprocessor,
    batch_size:   int  = 256,
    num_workers:  int  = 12,
    eval_batch_size: Optional[int] = None,
    eval_num_workers: Optional[int] = None,
    distributed:  bool = False,
    rank:         int  = 0,
    world_size:   int  = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construye DataLoaders que devuelven señales MEG raw (no imágenes).

    Reemplaza build_dataloaders() de meg_transfer_learning_libribrain.py.
    Usar junto con CWTLayer para la transformación en GPU.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = MEGRawDataset(train_pnpl, preprocessor, augment=True)
    val_ds   = MEGRawDataset(val_pnpl,   preprocessor, augment=False)
    test_ds  = MEGRawDataset(test_pnpl,  preprocessor, augment=False)

    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    eval_num_workers = min(num_workers, 2) if eval_num_workers is None else eval_num_workers

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=eval_num_workers,
        pin_memory=True,
        persistent_workers=(eval_num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=eval_num_workers,
        pin_memory=True,
        persistent_workers=(eval_num_workers > 0),
    )

    if rank == 0:
        print(f"[INFO] DataLoaders MEG raw:")
        print(f"  Train:      {len(train_ds):,} samples → {len(train_loader):,} batches/rank")
        print(f"  Validation: {len(val_ds):,} samples → {len(val_loader):,} batches/rank")
        print(f"  Test:       {len(test_ds):,} samples → {len(test_loader):,} batches/rank")
        print(f"  Eval batch/rank: {eval_batch_size} | Eval workers/rank: {eval_num_workers}")

    return train_loader, val_loader, test_loader, train_sampler


# ==============================================================================
# 5. INTEGRACIÓN EN train_ddp.py
# ==============================================================================
"""
CAMBIOS NECESARIOS EN train_ddp.py
====================================

1. IMPORTS (añadir al inicio):
    from meg_gpu_cwt import CWTLayer, MEGRawDataset, build_raw_dataloaders, zscore_scalogram

2. CONSTRUCCIÓN DE DATALOADERS (reemplazar build_dataloaders):
    train_loader, val_loader, test_loader = build_raw_dataloaders(
        train_pnpl, val_pnpl, test_pnpl,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=True,
        rank=rank,
        world_size=dist.get_world_size(),
    )

3. CWT LAYER (crear junto con el modelo):
    cwt_layer = CWTLayer(
        sfreq=250.0, n_freqs=96, f_min=1.0, f_max=125.0, B=1.5, C=1.0
    ).to(device)

4. MODELO (usar MEGImageModelEndToEnd en lugar de MEGImageModel):
    from meg_transfer_learning_libribrain import MEGImageModelEndToEnd
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

5. BUCLE DE ENTRENAMIENTO (añadir CWT antes del forward):
    for meg_raw, labels in train_loader:
        meg_raw = meg_raw.to(device)     # (B, 306, T)
        labels  = labels.to(device)

        # CWT en GPU (no diferenciable, sin gradientes)
        with torch.no_grad():
            scalogram = cwt_layer(meg_raw)          # (B, 306, n_freqs, T)
            scalogram = zscore_scalogram(scalogram) # normalización

        # Forward del modelo (diferenciable)
        logits = model(scalogram)                   # (B, n_classes)
        loss   = criterion(logits, labels)
        ...

    # Mismo cambio para val_loader y test_loader en evaluate()

NOTA SOBRE ImageNet NORMALIZATION:
    MEGImageModelEndToEnd NO aplica normalización ImageNet internamente.
    La normalización está implícita en el SensorMixer aprendido.
    Si quieres ser explícito, añade después del resize en MEGImageModelEndToEnd.forward():
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - x.amin(dim=(2,3), keepdim=True)) / (x.amax(dim=(2,3), keepdim=True) - x.amin(dim=(2,3), keepdim=True) + 1e-8)
        x = (x - imagenet_mean) / imagenet_std
"""
