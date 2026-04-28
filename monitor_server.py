#!/usr/bin/env python3
"""
================================================================================
  monitor_server.py — Dashboard de monitorización del sweep MEG
================================================================================
  Servidor HTTP autocontenido (stdlib pura, sin dependencias).
  Sirve un dashboard en tiempo real leyendo los mismos volúmenes que el training.

  Lanzamiento standalone:
    python3 monitor_server.py --port 8080 --base-dir /ruta/al/proyecto

  Via docker compose (ver docker-compose.yml):
    docker compose up monitor

  Acceso:
    http://<ip-servidor>:8080

  API JSON (para scripting):
    GET /api/status          → estado global del sweep
    GET /api/log?exp=<name>  → últimas líneas del log de un experimento
================================================================================
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── Configuración global (se sobreescribe con args) ────────────────────────────
BASE_DIR     = Path(os.environ.get("BASE_DIR", "/workspace"))
LOGS_DIR     = BASE_DIR / "logs"
RESULTS_DIR  = BASE_DIR / "results"
CKPT_DIR     = BASE_DIR / "checkpoints"
PLAN_FILE    = BASE_DIR / ".sweep_plan.json"
REFRESH_SECS = 8

# Espacio clásico (fallback cuando no hay sweep_mode explícito)
TASKS      = ["phoneme", "speech"]
BACKBONES  = ["resnet18", "efficientnet_b0", "vit_tiny"]
STRATEGIES = ["frozen", "partial_ft"]
CLASSIC_EXPS = [
    f"{t}__{b}__{s}"
    for t in TASKS
    for b in BACKBONES
    for s in STRATEGIES
]
PRECOMPUTE_TASKS = TASKS[:]


def load_sweep_plan() -> dict:
    if not PLAN_FILE.exists():
        return {}
    try:
        with PLAN_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_sweep_mode() -> str:
    plan = load_sweep_plan()
    mode = plan.get("mode") if isinstance(plan, dict) else None
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()

    mode_file = BASE_DIR / ".sweep_mode"
    if not mode_file.exists():
        return "classic"
    try:
        mode = mode_file.read_text(encoding="utf-8").strip().lower()
        return mode or "classic"
    except Exception:
        return "classic"


def _parse_iso_ts(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def _resolve_plan_path(raw_path: str | None) -> Path | None:
    if not raw_path or not isinstance(raw_path, str):
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else BASE_DIR / path


def _normalize_plan_experiment_entries(plan: dict) -> dict[str, dict]:
    experiments = plan.get("experiments", []) if isinstance(plan, dict) else []
    normalized: dict[str, dict] = {}
    if not isinstance(experiments, list):
        return normalized

    for item in experiments:
        if isinstance(item, str):
            normalized[item] = {"name": item}
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name:
                normalized[name] = item
    return normalized


def _normalize_plan_precompute_entries(plan: dict, stage_key: str) -> dict[str, dict]:
    precompute = plan.get("precompute", {}) if isinstance(plan, dict) else {}
    stage_items = precompute.get(stage_key, []) if isinstance(precompute, dict) else []
    normalized: dict[str, dict] = {}
    if not isinstance(stage_items, list):
        return normalized

    for item in stage_items:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        if isinstance(task, str) and task:
            normalized[task] = item
    return normalized


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


EXP_HEADER_RE = re.compile(r"━━━ \[\d+/\d+\] (.+?) ━━━")
PRECOMPUTE_LAUNCH_RE = re.compile(r"Lanzando precompute_stats para task='([^']+)'")
PRECOMPUTE_SKIP_RE = re.compile(r"Stats para '([^']+)' ya calculadas")
PRECOMPUTE_DONE_RE = re.compile(r"Precompute '([^']+)' completado")
PRECOMPUTE_FAIL_RE = re.compile(r"Precompute '([^']+)' falló")
RUN_TS_RE = re.compile(r"(\d{8}_\d{6})")


def _extract_run_ts(raw_value: str | Path | None) -> str | None:
    if raw_value is None:
        return None
    text = raw_value.name if isinstance(raw_value, Path) else str(raw_value)
    match = RUN_TS_RE.search(text)
    return match.group(1) if match else None


def _parse_run_ts(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S").timestamp()
    except Exception:
        return None


def _latest_mode_log_path(mode: str, kind: str) -> Path | None:
    path = LOGS_DIR / f"latest_{mode}_{kind}.log"
    return path.resolve() if path.exists() else None


def _infer_active_run_ts(plan: dict | None = None, mode: str | None = None) -> str | None:
    if isinstance(plan, dict):
        run_ts = plan.get("run_ts")
        if isinstance(run_ts, str) and run_ts.strip():
            return run_ts.strip()

        run_ts = _extract_run_ts(plan.get("sweep_log"))
        if run_ts:
            return run_ts

    mode = mode or get_sweep_mode()
    for kind in ("sweep", "coordinator"):
        path = _latest_mode_log_path(mode, kind)
        run_ts = _extract_run_ts(path)
        if run_ts:
            return run_ts

    for path in sorted(
        (p for p in LOGS_DIR.glob("sweep_*.log") if not p.name.startswith("sweep_coordinator_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        run_ts = _extract_run_ts(path)
        if run_ts:
            return run_ts

    return None


def _parse_sweep_log_statuses(sweep_log_path: Path | None) -> dict:
    statuses = {
        "experiments": {},
        "precompute": {"stats": {}, "images": {}},
    }
    if sweep_log_path is None or not sweep_log_path.exists():
        return statuses

    current_exp = None
    try:
        with sweep_log_path.open(encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = _strip_ansi(raw_line).strip()
                if not line:
                    continue

                m = EXP_HEADER_RE.search(line)
                if m:
                    current_exp = m.group(1).strip()
                    statuses["experiments"].setdefault(current_exp, {"status": "pending"})
                    continue

                m = PRECOMPUTE_LAUNCH_RE.search(line)
                if m:
                    statuses["precompute"]["stats"][m.group(1)] = {"status": "running"}
                    continue

                m = PRECOMPUTE_SKIP_RE.search(line)
                if m:
                    statuses["precompute"]["stats"][m.group(1)] = {"status": "skipped"}
                    continue

                m = PRECOMPUTE_DONE_RE.search(line)
                if m:
                    statuses["precompute"]["stats"][m.group(1)] = {"status": "done"}
                    continue

                m = PRECOMPUTE_FAIL_RE.search(line)
                if m:
                    statuses["precompute"]["stats"][m.group(1)] = {"status": "failed"}
                    continue

                if current_exp is None:
                    continue

                if "Ya completado (sentinel existente). Saltando." in line:
                    statuses["experiments"][current_exp] = {"status": "skipped"}
                elif "Contenedor:" in line:
                    statuses["experiments"][current_exp] = {"status": "running"}
                elif "Completado en" in line:
                    statuses["experiments"][current_exp] = {"status": "done"}
                elif "FALLIDO" in line:
                    statuses["experiments"][current_exp] = {"status": "failed"}
    except Exception:
        return statuses

    return statuses


def _get_active_sweep_log_path(plan: dict | None = None) -> Path | None:
    mode = get_sweep_mode()

    latest_mode_path = _latest_mode_log_path(mode, "sweep")
    if latest_mode_path:
        return latest_mode_path

    if isinstance(plan, dict):
        run_ts = _infer_active_run_ts(plan, mode)
        if run_ts:
            run_path = LOGS_DIR / f"sweep_{run_ts}.log"
            if run_path.exists():
                return run_path

        planned_path = _resolve_plan_path(plan.get("sweep_log"))
        if planned_path and planned_path.exists():
            return planned_path

    sweep_logs = sorted(
        (
            p for p in LOGS_DIR.glob("sweep_*.log")
            if p.name.startswith("sweep_") and not p.name.startswith("sweep_coordinator_")
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return sweep_logs[0] if sweep_logs else None


def _get_active_coordinator_log_path(plan: dict | None = None) -> Path | None:
    mode = get_sweep_mode()

    latest_mode_path = _latest_mode_log_path(mode, "coordinator")
    if latest_mode_path:
        return latest_mode_path

    if isinstance(plan, dict):
        run_ts = _infer_active_run_ts(plan, mode)
        if run_ts:
            run_path = LOGS_DIR / f"sweep_coordinator_{mode}_{run_ts}.log"
            if run_path.exists():
                return run_path

    coordinator_logs = sorted(
        LOGS_DIR.glob(f"sweep_coordinator_{mode}_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if coordinator_logs:
        return coordinator_logs[0]

    coordinator_logs = sorted(
        LOGS_DIR.glob("sweep_coordinator_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return coordinator_logs[0] if coordinator_logs else None


def _get_active_run_started_at(
    plan: dict | None = None,
    sweep_log_path: Path | None = None,
    coordinator_log_path: Path | None = None,
) -> float | None:
    mode = get_sweep_mode()
    run_ts = _infer_active_run_ts(plan, mode)
    started_at = _parse_run_ts(run_ts)
    if started_at is not None:
        return started_at

    for path in (sweep_log_path, coordinator_log_path):
        if path and path.exists():
            try:
                return min(path.stat().st_ctime, path.stat().st_mtime)
            except Exception:
                pass

    if isinstance(plan, dict):
        return _parse_iso_ts(plan.get("generated_at"))

    return None


def get_sweep_context() -> dict:
    plan = load_sweep_plan()
    sweep_log_path = _get_active_sweep_log_path(plan)
    coordinator_log_path = _get_active_coordinator_log_path(plan)
    run_started_at = _get_active_run_started_at(plan, sweep_log_path, coordinator_log_path)
    return {
        "plan": plan,
        "run_ts": _infer_active_run_ts(plan),
        "run_started_at": run_started_at,
        "experiments": _normalize_plan_experiment_entries(plan),
        "precompute_stats": _normalize_plan_precompute_entries(plan, "stats"),
        "precompute_images": _normalize_plan_precompute_entries(plan, "images"),
        "log_statuses": _parse_sweep_log_statuses(sweep_log_path),
        "sweep_log_path": sweep_log_path,
        "coordinator_log_path": coordinator_log_path,
    }


def _artifact_belongs_to_current_run(path: Path, started_at: float | None, slack_secs: float = 1.0) -> bool:
    if not path.exists():
        return False
    if started_at is None:
        return True
    try:
        stat = path.stat()
        freshest = max(stat.st_mtime, stat.st_ctime)
        return freshest >= started_at - slack_secs
    except Exception:
        return False


def discover_experiments() -> list[str]:
    """
    Descubre experimentos automáticamente para soportar:
      - modo clásico: task__backbone__strategy
      - modo speech-image: speech_image__<exp_id>
    """
    plan = load_sweep_plan()
    planned_entries = _normalize_plan_experiment_entries(plan)
    if planned_entries:
        return list(planned_entries.keys())

    mode = get_sweep_mode()
    if mode == "classic":
        return CLASSIC_EXPS[:]

    # speech_image y otros modos: descubrir por sentinels, logs y results
    exps = set()

    for p in BASE_DIR.glob(".exp_done_*"):
        exps.add(p.name.replace(".exp_done_", "", 1))
    for p in LOGS_DIR.glob("*.log"):
        n = p.stem
        if n.startswith("sweep_") or n.startswith("precompute_"):
            continue
        exps.add(n)
    for p in RESULTS_DIR.glob("*/final_results.json"):
        exps.add(p.parent.name)

    return sorted(exps)


# ==============================================================================
# LÓGICA DE ESTADO
# ==============================================================================

def _tail_last_line(path: Path, max_bytes: int = 1024) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode("utf-8", errors="replace").strip()
        return tail.split("\n")[-1][:180]
    except Exception:
        return ""


def _read_recent_log(path: Path | None, *, max_bytes: int = 2000, lines: int | None = None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            content = f.read().decode("utf-8", errors="replace")
        if lines is not None:
            content = "\n".join(content.split("\n")[-lines:])
        return content
    except Exception:
        return ""


def _extract_exp_section_from_log(path: Path | None, exp: str, *, lines: int = 80) -> str:
    if path is None or not path.exists():
        return ""

    captured: list[str] = []
    capturing = False
    target_header_suffix = f" {exp} ━━━"

    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                stripped = _strip_ansi(raw_line).strip()
                is_exp_header = bool(EXP_HEADER_RE.search(stripped))

                if is_exp_header and stripped.endswith(target_header_suffix):
                    capturing = True
                    captured = [raw_line.rstrip("\n")]
                    continue

                if capturing and is_exp_header:
                    break

                if capturing:
                    captured.append(raw_line.rstrip("\n"))
    except Exception:
        return ""

    if not captured:
        return ""

    return "\n".join(captured[-lines:])


def _get_fallback_exp_log(exp: str, ctx: dict, *, lines: int = 80) -> str:
    for path in (ctx.get("sweep_log_path"), ctx.get("coordinator_log_path")):
        content = _extract_exp_section_from_log(path, exp, lines=lines)
        if content.strip():
            return content
    return ""


def _get_stage_status_for_task(
    task: str,
    sentinel_name: str,
    log_name: str,
    *,
    plan_status: str | None = None,
    started_at: float | None = None,
    log_override: str | None = None,
) -> dict:
    """
    Estado por tarea para etapas de precompute.
    status: pending | running | done | failed | skipped
    """
    sentinel = BASE_DIR / sentinel_name.format(task=task)
    log_path = LOGS_DIR / log_name.format(task=task)

    status = "pending"
    elapsed_min = None
    tail_line = _tail_last_line(log_path)
    tail_lower = tail_line.lower()
    done_markers = (
        "precompute finalizado",
        "completado",
        "done",
    )

    if log_override in {"pending", "running", "done", "failed", "skipped"}:
        status = log_override
    elif plan_status in {"pending", "running", "done", "failed", "skipped"}:
        status = plan_status

    if status == "pending" and _artifact_belongs_to_current_run(sentinel, started_at):
        status = "done"
        if log_path.exists():
            try:
                elapsed_min = int((time.time() - log_path.stat().st_ctime) / 60)
            except Exception:
                elapsed_min = None
    elif status == "pending" and log_path.exists() and _artifact_belongs_to_current_run(log_path, started_at):
        try:
            age_min = (time.time() - log_path.stat().st_mtime) / 60
            if any(marker in tail_lower for marker in done_markers):
                status = "done"
            else:
                status = "running" if age_min < 10 else "failed"
            elapsed_min = int((time.time() - log_path.stat().st_ctime) / 60)
        except Exception:
            status = "failed"

    return {
        "task": task,
        "status": status,
        "elapsed_min": elapsed_min,
        "last_line": tail_line,
        "log_path": str(log_path),
    }


def get_precompute_status(ctx: dict | None = None) -> dict:
    """
    Estado agregado de precompute de stats e imágenes por tarea.
    """
    ctx = ctx or get_sweep_context()
    plan = ctx["plan"]
    precompute_plan = plan.get("precompute", {}) if isinstance(plan, dict) else {}
    rich_stats = ctx["precompute_stats"]
    rich_images = ctx["precompute_images"]
    log_precompute = ctx["log_statuses"].get("precompute", {})
    log_stats = log_precompute.get("stats", {}) if isinstance(log_precompute, dict) else {}
    log_images = log_precompute.get("images", {}) if isinstance(log_precompute, dict) else {}
    started_at = ctx["run_started_at"]

    if rich_stats or rich_images:
        stats_tasks = list(rich_stats.keys())
        images_tasks = list(rich_images.keys())
    elif isinstance(precompute_plan, dict):
        stats_tasks = [str(t) for t in precompute_plan.get("stats_tasks", [])]
        images_tasks = [str(t) for t in precompute_plan.get("images_tasks", [])]
    elif get_sweep_mode() == "classic":
        stats_tasks = PRECOMPUTE_TASKS[:]
        images_tasks = PRECOMPUTE_TASKS[:]
    else:
        stats_tasks = []
        images_tasks = []

    stats = [
        _get_stage_status_for_task(
            t,
            sentinel_name=".precompute_done_{task}",
            log_name="precompute_{task}.log",
            plan_status=rich_stats.get(t, {}).get("status"),
            started_at=started_at,
            log_override=log_stats.get(t, {}).get("status"),
        )
        for t in stats_tasks
    ]
    images = [
        _get_stage_status_for_task(
            t,
            sentinel_name=".precompute_images_done_{task}",
            log_name="precompute_images_{task}.log",
            plan_status=rich_images.get(t, {}).get("status"),
            started_at=started_at,
            log_override=log_images.get(t, {}).get("status"),
        )
        for t in images_tasks
    ]

    def _counts(items):
        c = {"done": 0, "running": 0, "failed": 0, "pending": 0, "skipped": 0}
        for it in items:
            c[it["status"]] += 1
        return c

    return {
        "tasks": sorted(set(stats_tasks + images_tasks)),
        "stats": stats,
        "images": images,
        "counts_stats": _counts(stats),
        "counts_images": _counts(images),
        "show_stats_stage": bool(stats_tasks),
        "show_images_stage": bool(images_tasks),
    }


def get_gpu_info() -> list[dict]:
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_info = []
        try:
            for idx in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                gpu_info.append({
                    "index": str(idx),
                    "name": str(name),
                    "util": str(getattr(util, "gpu", 0)),
                    "mem_used": str(int(mem.used / 1024 / 1024)),
                    "mem_total": str(int(mem.total / 1024 / 1024)),
                    "temp": str(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                })
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        if gpu_info:
            return gpu_info
    except Exception:
        pass

    gpu_info = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode().strip()
        for line in out.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu_info.append({
                    "index": parts[0], "name": parts[1],
                    "util": parts[2], "mem_used": parts[3],
                    "mem_total": parts[4], "temp": parts[5],
                })
    except Exception:
        pass
    return gpu_info


def get_exp_status(exp: str, ctx: dict | None = None) -> dict:
    """
    Determina el estado de un experimento a partir de los archivos en disco.
    Returns: dict con status, epoch_current, epoch_total, f1, bal_acc, elapsed_min
    """
    ctx = ctx or get_sweep_context()
    plan_entry = ctx["experiments"].get(exp, {})
    log_override = ctx["log_statuses"].get("experiments", {}).get(exp, {})
    started_at = ctx["run_started_at"]

    done_sentinel  = BASE_DIR / f".exp_done_{exp}"
    job_log        = LOGS_DIR / f"{exp}.log"
    training_state = CKPT_DIR / exp / "training_state.json"
    final_results  = RESULTS_DIR / exp / "final_results.json"

    result = {
        "exp": exp,
        "status": "pending",       # pending | running | done | failed | skipped
        "epoch_current": None,
        "epoch_total": None,
        "f1_macro": None,
        "balanced_acc": None,
        "best_val_f1": None,
        "elapsed_min": None,
        "last_line": "",
        "log_mtime": None,
    }

    if plan_entry.get("status") in {"pending", "running", "done", "failed", "skipped"}:
        result["status"] = plan_entry["status"]
    if log_override.get("status") in {"pending", "running", "done", "failed", "skipped"}:
        result["status"] = log_override["status"]

    current_results = _artifact_belongs_to_current_run(final_results, started_at)
    current_done_sentinel = _artifact_belongs_to_current_run(done_sentinel, started_at)
    current_job_log = _artifact_belongs_to_current_run(job_log, started_at)
    current_training_state = _artifact_belongs_to_current_run(training_state, started_at)

    if result["status"] == "skipped":
        return result

    # ── Completado ────────────────────────────────────────────────────────────
    # `final_results.json` es la evidencia más fiable de finalización.
    if result["status"] == "done" or current_results or current_done_sentinel:
        result["status"] = "done"
        try:
            with open(final_results) as f:
                d = json.load(f)
            result["f1_macro"]    = d.get("test_f1_macro")
            result["balanced_acc"] = d.get("test_balanced_acc")
            result["best_val_f1"] = d.get("best_val_f1")
        except Exception:
            pass
        return result

    # ── En curso: el log existe y tiene actividad reciente ────────────────────
    if current_job_log:
        try:
            mtime = job_log.stat().st_mtime
            result["log_mtime"] = mtime
            age_min = (time.time() - mtime) / 60
            # Si el log se modificó hace menos de 10 min, está corriendo
            if result["status"] != "running":
                result["status"] = "running" if age_min < 10 else "failed"
            elif age_min >= 10:
                result["status"] = "failed"

            # Leer training_state.json para epoch actual
            if current_training_state:
                with open(training_state) as f:
                    ts = json.load(f)
                result["epoch_current"] = ts.get("epoch")
                result["best_val_f1"]   = ts.get("metrics", {}).get("val_f1_macro")

            # Última línea del log
            with open(job_log, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 512))
                tail = f.read().decode("utf-8", errors="replace").strip()
                result["last_line"] = tail.split("\n")[-1][:120]

            # Tiempo transcurrido (desde creación del log)
            ctime = job_log.stat().st_ctime
            result["elapsed_min"] = int((time.time() - ctime) / 60)

        except Exception:
            pass

    return result


def get_sweep_status() -> dict:
    """Estado global del sweep."""
    ctx = get_sweep_context()
    all_exps = discover_experiments()
    exps = [get_exp_status(e, ctx=ctx) for e in all_exps]
    precompute = get_precompute_status(ctx=ctx)
    counts = {"done": 0, "running": 0, "failed": 0, "pending": 0, "skipped": 0}
    for e in exps:
        counts[e["status"]] += 1

    completed = [e for e in exps if e["status"] == "done" and e["f1_macro"] is not None]
    completed.sort(key=lambda x: x["f1_macro"] or 0, reverse=True)

    # Leer el log del sweep activo. Con `--detach` preferimos el sweep log y,
    # si todavía no tiene contenido útil, caemos al coordinator log.
    sweep_tail = _read_recent_log(ctx["sweep_log_path"], max_bytes=2000)
    if not sweep_tail.strip():
        sweep_tail = _read_recent_log(ctx["coordinator_log_path"], max_bytes=2000)

    return {
        "timestamp": datetime.now().isoformat(),
        "total": len(all_exps),
        "counts": counts,
        "precompute": precompute,
        "experiments": exps,
        "leaderboard": completed,
        "sweep_tail": sweep_tail,
        "gpu_info": get_gpu_info(),
        "sweep_mode": get_sweep_mode(),
    }


def get_exp_log(exp: str, lines: int = 80, ctx: dict | None = None) -> str:
    """Últimas N líneas del log de un experimento del sweep activo."""
    ctx = ctx or get_sweep_context()
    log_path = LOGS_DIR / f"{exp}.log"
    if log_path.exists() and _artifact_belongs_to_current_run(log_path, ctx["run_started_at"]):
        content = _read_recent_log(log_path, max_bytes=lines * 200, lines=lines)
        if content:
            return content

    fallback = _get_fallback_exp_log(exp, ctx, lines=lines)
    if fallback:
        return fallback

    return f"[Sin log del sweep actual para {exp}]"


# ==============================================================================
# HTML DEL DASHBOARD
# ==============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MEG Sweep Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg:        #0b0d12;
    --surface:   #12151e;
    --border:    #1e2433;
    --accent:    #00d4a8;
    --accent2:   #7c6bff;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --done:      #00d4a8;
    --running:   #7c6bff;
    --failed:    #ef4444;
    --skipped:   #f59e0b;
    --pending:   #334155;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Syne', sans-serif;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 100;
  }
  .logo {
    font-family: var(--sans);
    font-weight: 800;
    font-size: 18px;
    letter-spacing: -0.5px;
  }
  .logo span { color: var(--accent); }
  .header-meta {
    display: flex;
    align-items: center;
    gap: 20px;
    color: var(--muted);
    font-size: 11px;
  }
  .pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s infinite;
    display: inline-block;
    margin-right: 6px;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
  }

  /* ── Layout ── */
  main { padding: 24px 32px; max-width: 1600px; margin: 0 auto; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  @media (max-width: 1100px) { .grid-2 { grid-template-columns: 1fr; } }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }
  .card-title {
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
  }

  /* ── Precompute strip ── */
  .precompute-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  @media (max-width: 1100px) { .precompute-grid { grid-template-columns: 1fr; } }

  .precompute-stage {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    background: rgba(255,255,255,0.015);
  }
  .precompute-stage-title {
    font-size: 10px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .precompute-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 0;
    border-top: 1px solid rgba(30,36,51,0.4);
  }
  .precompute-row:first-of-type { border-top: none; }
  .precompute-task {
    font-size: 12px;
    color: var(--text);
    text-transform: lowercase;
  }
  .status-pill {
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 7px;
    border-radius: 3px;
    font-weight: 700;
  }
  .pill-done    { background: rgba(0,212,168,0.15); color: var(--done); }
  .pill-running { background: rgba(124,107,255,0.15); color: var(--running); }
  .pill-failed  { background: rgba(239,68,68,0.15); color: var(--failed); }
  .pill-skipped { background: rgba(245,158,11,0.18); color: var(--skipped); }
  .pill-pending { background: rgba(51,65,85,0.35); color: var(--muted); }

  /* ── Stat boxes ── */
  .stats-row { display: flex; gap: 16px; margin-bottom: 20px; }
  .stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 24px;
    flex: 1;
    text-align: center;
  }
  .stat-num {
    font-family: var(--sans);
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-label { font-size: 11px; color: var(--muted); letter-spacing: 1px; }
  .stat-done    .stat-num { color: var(--done); }
  .stat-running .stat-num { color: var(--running); }
  .stat-failed  .stat-num { color: var(--failed); }
  .stat-skipped .stat-num { color: var(--skipped); }
  .stat-pending .stat-num { color: var(--muted); }

  /* ── Progress bar ── */
  .progress-wrap { margin-bottom: 20px; }
  .progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .progress-bar {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 3px;
    transition: width 1s ease;
  }

  /* ── Experiment grid ── */
  .exp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
  }
  .exp-card {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
    overflow: hidden;
  }
  .exp-card:hover { border-color: var(--accent2); background: rgba(124,107,255,0.05); }
  .exp-card.selected { border-color: var(--accent); background: rgba(0,212,168,0.05); }

  .exp-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
  }
  .exp-card.done    ::before { background: var(--done); }
  .exp-card.running ::before { background: var(--running); animation: pulse 1.5s infinite; }
  .exp-card.failed  ::before { background: var(--failed); }
  .exp-card.skipped ::before { background: var(--skipped); }
  .exp-card.pending ::before { background: var(--pending); }

  .exp-name { font-size: 11px; font-weight: 600; margin-bottom: 6px; color: var(--text); }
  .exp-badge {
    display: inline-block;
    font-size: 9px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .badge-done    { background: rgba(0,212,168,0.15);   color: var(--done);    }
  .badge-running { background: rgba(124,107,255,0.15); color: var(--running); }
  .badge-failed  { background: rgba(239,68,68,0.15);   color: var(--failed);  }
  .badge-skipped { background: rgba(245,158,11,0.18);  color: var(--skipped); }
  .badge-pending { background: rgba(51,65,85,0.3);     color: var(--muted);   }

  .exp-metric { font-size: 11px; color: var(--muted); }
  .exp-metric strong { color: var(--accent); font-size: 13px; }
  .exp-epoch  { font-size: 10px; color: var(--muted); margin-top: 4px; }
  .epoch-bar {
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    margin-top: 6px;
    overflow: hidden;
  }
  .epoch-fill {
    height: 100%;
    background: var(--running);
    border-radius: 1px;
  }

  /* ── Log viewer ── */
  .log-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .log-select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 4px;
    outline: none;
    cursor: pointer;
  }
  .log-box {
    background: #070910;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    font-size: 11px;
    line-height: 1.7;
    color: #94a3b8;
    height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    font-family: var(--mono);
  }
  .log-box .log-epoch  { color: #e2e8f0; }
  .log-box .log-best   { color: var(--accent); font-weight: 700; }
  .log-box .log-warn   { color: var(--warn); }
  .log-box .log-error  { color: var(--failed); }
  .log-box .log-info   { color: #60a5fa; }

  /* ── Leaderboard ── */
  table { width: 100%; border-collapse: collapse; }
  th {
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(30,36,51,0.5);
    font-size: 12px;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .rank { color: var(--muted); font-size: 11px; }
  .rank-1 { color: var(--accent); font-weight: 700; }
  .metric-val { color: var(--accent); font-weight: 600; }
  .task-badge {
    display: inline-block;
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 3px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .task-phoneme { background: rgba(124,107,255,0.15); color: var(--accent2); }
  .task-speech  { background: rgba(0,212,168,0.15);   color: var(--accent);  }

  /* ── GPU strip ── */
  .gpu-strip { display: flex; gap: 12px; margin-bottom: 20px; }
  .gpu-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
  }
  .gpu-name { font-size: 11px; color: var(--muted); margin-bottom: 8px; }
  .gpu-util-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 6px; }
  .gpu-util-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent2), var(--accent)); transition: width 1s; }
  .gpu-stats { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); }
  .gpu-stats span { color: var(--text); }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    width: 10px; height: 10px;
    border: 2px solid var(--border);
    border-top-color: var(--running);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 6px;
    vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #last-update { color: var(--muted); font-size: 10px; }
</style>
</head>
<body>

<header>
  <div class="logo">MEG<span>·</span>SWEEP</div>
  <div class="header-meta">
    <span><span class="pulse"></span>Auto-refresh cada REFRESH_SECSs</span>
    <span id="last-update">—</span>
  </div>
</header>

<main>
  <!-- GPU strip -->
  <div id="gpu-strip" class="gpu-strip"></div>

  <!-- Precompute status -->
  <div class="card" id="precompute-card" style="margin-bottom:20px;">
    <div class="card-title">Precompute</div>
    <div class="precompute-grid">
      <div class="precompute-stage" id="precompute-stats-stage">
        <div class="precompute-stage-title">Stats (H5 normalización)</div>
        <div id="precompute-stats-wrap">
          <div class="precompute-row"><span class="precompute-task">cargando…</span></div>
        </div>
      </div>
      <div class="precompute-stage" id="precompute-images-stage">
        <div class="precompute-stage-title">Imágenes (señal + augmentación + CWT)</div>
        <div id="precompute-images-wrap">
          <div class="precompute-row"><span class="precompute-task">cargando…</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Stats row -->
  <div class="stats-row">
    <div class="stat-box stat-done">
      <div class="stat-num" id="cnt-done">—</div>
      <div class="stat-label">COMPLETADOS</div>
    </div>
    <div class="stat-box stat-running">
      <div class="stat-num" id="cnt-running">—</div>
      <div class="stat-label">EN CURSO</div>
    </div>
    <div class="stat-box stat-failed">
      <div class="stat-num" id="cnt-failed">—</div>
      <div class="stat-label">FALLIDOS</div>
    </div>
    <div class="stat-box stat-skipped">
      <div class="stat-num" id="cnt-skipped">—</div>
      <div class="stat-label">SALTADOS</div>
    </div>
    <div class="stat-box stat-pending">
      <div class="stat-num" id="cnt-pending">—</div>
      <div class="stat-label">PENDIENTES</div>
    </div>
  </div>

  <!-- Progress bar -->
  <div class="progress-wrap">
    <div class="progress-label">
      <span>Progreso del sweep</span>
      <span id="progress-text">0 / 12</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
  </div>

  <!-- Experiment grid -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">Experimentos</div>
    <div class="exp-grid" id="exp-grid"></div>
  </div>

  <div class="grid-2">
    <!-- Log viewer -->
    <div class="card">
      <div class="log-header">
        <div class="card-title" style="margin-bottom:0">Log en tiempo real</div>
        <select class="log-select" id="log-select" onchange="selectExp(this.value)">
          <option value="__sweep__">— sweep global —</option>
        </select>
      </div>
      <div class="log-box" id="log-box">Cargando...</div>
    </div>

    <!-- Leaderboard -->
    <div class="card">
      <div class="card-title">Leaderboard (test F1-macro)</div>
      <div id="leaderboard-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Experimento</th><th>Task</th>
              <th>F1-macro</th><th>Bal. Acc</th><th>Best Val F1</th>
            </tr>
          </thead>
          <tbody id="leaderboard-body">
            <tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px">
              Sin resultados todavía...
            </td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</main>

<script>
const REFRESH = REFRESH_SECS * 1000;
let selectedExp = '__sweep__';
let autoPickedRunningExp = false;

function selectExp(exp) {
  selectedExp = exp;
  document.querySelectorAll('.exp-card').forEach(c => c.classList.remove('selected'));
  const card = document.querySelector(`.exp-card[data-exp="${exp}"]`);
  if (card) card.classList.add('selected');
  fetchLog();
}

function colorLog(text) {
  return text
    .replace(/^(Epoch \d+\/\d+.*)/gm, '<span class="log-epoch">$1</span>')
    .replace(/(★ BEST)/g, '<span class="log-best">$1</span>')
    .replace(/(\[WARN\].*)/g, '<span class="log-warn">$1</span>')
    .replace(/(\[ERROR\].*|FAILED|Error)/g, '<span class="log-error">$1</span>')
    .replace(/(\[INFO\].*|\[DDP\].*|\[Checkpoint\].*)/g, '<span class="log-info">$1</span>');
}

async function fetchLog() {
  const url = selectedExp === '__sweep__'
    ? '/api/log?exp=__sweep__'
    : `/api/log?exp=${encodeURIComponent(selectedExp)}`;
  try {
    const r = await fetch(url);
    const text = await r.text();
    const box = document.getElementById('log-box');
    box.innerHTML = colorLog(text);
    box.scrollTop = box.scrollHeight;
  } catch(e) {}
}

async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    render(d);
    document.getElementById('last-update').textContent =
      'Actualizado: ' + new Date().toLocaleTimeString('es-ES');
  } catch(e) {
    console.error(e);
  }
}

function statusPill(status) {
  return `<span class="status-pill pill-${status}">${status}</span>`;
}

function renderPrecomputeStage(targetId, rows) {
  const wrap = document.getElementById(targetId);
  if (!rows || rows.length === 0) {
    wrap.innerHTML = '<div class="precompute-row"><span class="precompute-task">sin datos</span></div>';
    return;
  }
  wrap.innerHTML = rows.map(r => {
    const elapsed = r.elapsed_min != null ? ` · ${r.elapsed_min}m` : '';
    return `
      <div class="precompute-row" title="${(r.last_line || '').replace(/"/g, '&quot;')}">
        <span class="precompute-task">${r.task}${elapsed}</span>
        ${statusPill(r.status)}
      </div>
    `;
  }).join('');
}

function render(d) {
  // Precompute (stats + imágenes)
  if (d.precompute) {
    const statsVisible = !!d.precompute.show_stats_stage;
    const imagesVisible = !!d.precompute.show_images_stage;
    document.getElementById('precompute-card').style.display =
      (statsVisible || imagesVisible) ? 'block' : 'none';
    document.getElementById('precompute-stats-stage').style.display =
      statsVisible ? 'block' : 'none';
    document.getElementById('precompute-images-stage').style.display =
      imagesVisible ? 'block' : 'none';
    renderPrecomputeStage('precompute-stats-wrap', d.precompute.stats || []);
    renderPrecomputeStage('precompute-images-wrap', d.precompute.images || []);
  }

  // Counters
  document.getElementById('cnt-done').textContent    = d.counts.done;
  document.getElementById('cnt-running').textContent = d.counts.running;
  document.getElementById('cnt-failed').textContent  = d.counts.failed;
  document.getElementById('cnt-skipped').textContent = d.counts.skipped;
  document.getElementById('cnt-pending').textContent = d.counts.pending;

  // Progress
  const finished = d.counts.done + d.counts.failed + d.counts.skipped;
  const pct = d.total > 0 ? Math.round(finished / d.total * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = `${finished} / ${d.total}`;

  // GPU strip
  const gpuStrip = document.getElementById('gpu-strip');
  if (d.gpu_info && d.gpu_info.length > 0) {
    gpuStrip.innerHTML = d.gpu_info.map(g => `
      <div class="gpu-card">
        <div class="gpu-name">GPU ${g.index} — ${g.name}</div>
        <div class="gpu-util-bar">
          <div class="gpu-util-fill" style="width:${g.util}%"></div>
        </div>
        <div class="gpu-stats">
          <span>GPU <span>${g.util}%</span></span>
          <span>VRAM <span>${g.mem_used}/${g.mem_total} MB</span></span>
          <span>Temp <span>${g.temp}°C</span></span>
        </div>
      </div>`).join('');
    gpuStrip.style.display = 'flex';
  } else {
    gpuStrip.style.display = 'none';
  }

  // Experiment grid
  const grid = document.getElementById('exp-grid');
  const select = document.getElementById('log-select');
  const prevOptions = new Set(Array.from(select.options).map(o => o.value));

  grid.innerHTML = d.experiments.map(e => {
    const parts = e.exp.split('__');
    const shortName = parts.slice(1).join(' · ');
    const taskBadge = parts[0] === 'speech_image' ? 'speech' : parts[0];
    const epochInfo = e.epoch_current
      ? `<div class="exp-epoch">Epoch ${e.epoch_current}</div>
         <div class="epoch-bar"><div class="epoch-fill" style="width:${Math.round(e.epoch_current/50*100)}%"></div></div>`
      : '';
    const metric = e.f1_macro != null
      ? `<div class="exp-metric">F1 <strong>${e.f1_macro.toFixed(4)}</strong></div>`
      : e.best_val_f1 != null
        ? `<div class="exp-metric">Val F1 <strong>${e.best_val_f1.toFixed(4)}</strong></div>`
        : '';
    const running_icon = e.status === 'running' ? '<span class="spinner"></span>' : '';
    const sel = selectedExp === e.exp ? 'selected' : '';
    return `<div class="exp-card ${e.status} ${sel}" data-exp="${e.exp}" onclick="selectExp('${e.exp}')">
      <div class="exp-name">${running_icon}${taskBadge} · ${shortName}</div>
      <span class="exp-badge badge-${e.status}">${e.status}</span>
      ${metric}
      ${epochInfo}
    </div>`;
  }).join('');

  // Populate select with new running/done experiments
  d.experiments.forEach(e => {
    if (!prevOptions.has(e.exp) && ['running', 'done', 'failed'].includes(e.status)) {
      const opt = document.createElement('option');
      opt.value = e.exp;
      opt.textContent = e.exp;
      select.appendChild(opt);
    }
  });

  // Al abrir por primera vez, saltar automáticamente al experimento activo
  // más reciente para no quedarse mostrando un sweep antiguo.
  if (!autoPickedRunningExp && selectedExp === '__sweep__') {
    const running = d.experiments
      .filter(e => e.status === 'running')
      .sort((a, b) => (b.log_mtime || 0) - (a.log_mtime || 0));
    if (running.length > 0) {
      select.value = running[0].exp;
      selectExp(running[0].exp);
      autoPickedRunningExp = true;
    }
  }

  // Leaderboard
  const tbody = document.getElementById('leaderboard-body');
  if (d.leaderboard.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px">Sin resultados todavía...</td></tr>';
  } else {
    tbody.innerHTML = d.leaderboard.map((e, i) => {
      const parts = e.exp.split('__');
      const task = parts[0] === 'speech_image' ? 'speech' : parts[0];
      const name = parts.slice(1).join(' · ');
      const rankCls = i === 0 ? 'rank-1' : 'rank';
      const taskCls = `task-${task}`;
      return `<tr>
        <td class="${rankCls}">${i === 0 ? '🥇' : i + 1}</td>
        <td>${name}</td>
        <td><span class="task-badge ${taskCls}">${task}</span></td>
        <td class="metric-val">${(e.f1_macro || 0).toFixed(4)}</td>
        <td>${(e.balanced_acc || 0).toFixed(4)}</td>
        <td style="color:var(--muted)">${e.best_val_f1 != null ? e.best_val_f1.toFixed(4) : '—'}</td>
      </tr>`;
    }).join('');
  }
}

// ── Polling ─────────────────────────────────────────────────────────────────
fetchStatus();
fetchLog();
setInterval(fetchStatus, REFRESH);
setInterval(fetchLog, REFRESH);
</script>
</body>
</html>
"""

HTML = HTML_TEMPLATE.replace("REFRESH_SECS", str(REFRESH_SECS))


# ==============================================================================
# HTTP SERVER
# ==============================================================================

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Silenciar logs de acceso por defecto

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self.send_html(HTML)

        elif path == "/api/status":
            self.send_json(get_sweep_status())

        elif path == "/api/log":
            exp   = params.get("exp", ["__sweep__"])[0]
            lines = int(params.get("lines", [80])[0])
            ctx = get_sweep_context()
            if exp == "__sweep__":
                text = _read_recent_log(ctx["sweep_log_path"], max_bytes=lines * 200, lines=lines)
                if not text.strip():
                    text = _read_recent_log(ctx["coordinator_log_path"], max_bytes=lines * 200, lines=lines)
                if not text.strip():
                    text = "[Sin sweep log todavía]"
                self.send_text(text)
            else:
                self.send_text(get_exp_log(exp, lines, ctx=ctx))

        else:
            self.send_response(404)
            self.end_headers()


def main():
    global BASE_DIR, LOGS_DIR, RESULTS_DIR, CKPT_DIR
    parser = argparse.ArgumentParser(description="MEG Sweep Monitor")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    BASE_DIR    = Path(args.base_dir)
    LOGS_DIR    = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    CKPT_DIR    = BASE_DIR / "checkpoints"

    print(f"[Monitor] Leyendo datos desde: {BASE_DIR}")
    print(f"[Monitor] Dashboard disponible en: http://0.0.0.0:{args.port}")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Monitor] Parado.")


if __name__ == "__main__":
    main()
