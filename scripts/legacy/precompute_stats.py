#!/usr/bin/env python3
"""
================================================================================
  precompute_stats.py — Precálculo secuencial de stats de normalización H5
================================================================================

PROBLEMA QUE RESUELVE:
  pnpl abre los archivos H5 en modo "r+" para cachear mean/std por canal.
  En DDP, ambos procesos intentan hacer esto simultáneamente sobre los mismos
  archivos → BlockingIOError (errno=11, file lock conflict).

SOLUCIÓN:
  Ejecutar este script UNA SOLA VEZ antes del training DDP.
  Calcula las stats de forma secuencial (un solo proceso, sin conflictos de lock)
  y las escribe en cada H5. El training posterior sólo lee — sin "r+", sin locks.

USO:
  # Dentro del contenedor (lo lanza docker-compose automáticamente):
  python precompute_stats.py --data_path /workspace/libribrain_data --task phoneme

  # Fuera del contenedor, para debug:
  python precompute_stats.py --data_path ./libribrain_data --task phoneme

  # Ambas tareas de una vez:
  python precompute_stats.py --data_path ./libribrain_data --task phoneme
  python precompute_stats.py --data_path ./libribrain_data --task speech

CUÁNDO REPETIR:
  - Sólo si cambias de dataset o borras los archivos H5.
  - Si los H5 ya tienen las stats cacheadas, pnpl las detecta y el script
    termina rápido (no recalcula).
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

# Asegurar que el directorio del proyecto está en el path
sys.path.insert(0, str(Path(__file__).parent))

from meg_transfer_learning_libribrain import load_libribrain, LibriBrainConfig


def precompute(data_path: str, task: str) -> None:
    """
    Carga cada partición secuencialmente para forzar el cálculo y cacheo
    de las stats de normalización en los archivos H5.
    """
    partitions = ["train", "validation", "test"]

    print("=" * 70)
    print(f"  Precalculando stats de normalización")
    print(f"  Tarea:     {task}")
    print(f"  Data path: {data_path}")
    print("=" * 70)

    total_start = time.time()

    for partition in partitions:
        print(f"\n[precompute] {partition} ...", flush=True)
        t0 = time.time()

        load_libribrain(LibriBrainConfig(
            data_path=data_path,
            task=task,
            partition=partition,
            download=True,
        ))

        elapsed = time.time() - t0
        print(f"[precompute] ✓ {partition} listo ({elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print(f"  ✓ Stats calculadas para las {len(partitions)} particiones")
    print(f"  Tiempo total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Ya puedes lanzar: docker compose up meg_training_job")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Precalcula stats de normalización H5 para training DDP sin locks."
    )
    parser.add_argument(
        "--data_path",
        default="./libribrain_data",
        help="Ruta a los datos LibriBrain (default: ./libribrain_data)",
    )
    parser.add_argument(
        "--task",
        default="phoneme",
        choices=["phoneme", "speech"],
        help="Tarea a precalcular (default: phoneme)",
    )
    args = parser.parse_args()

    precompute(args.data_path, args.task)


if __name__ == "__main__":
    main()
