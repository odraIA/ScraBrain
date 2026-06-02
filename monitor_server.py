#!/usr/bin/env python3
"""
================================================================================
  monitor_server.py — Dashboard de monitorización de entrenamientos/evaluaciones MEG
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
    GET /api/status          → estado global de ejecuciones detectadas
    GET /api/log?exp=<name>  → últimas líneas del log de una ejecución
================================================================================
"""

import argparse
import glob
import json
import os
import re
import socket
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
DOCKER_SOCKET = Path(os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock"))
GENERIC_SCAN_ROOT_NAMES = ("logs", "checkpoints", "outputs", "multirun")
GENERIC_RUN_MARKERS = (
    "final_results.txt",
    "checkpoint_best.pt",
    "checkpoint_latest.pt",
    ".hydra/config.yaml",
)

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


def _safe_stat_mtime(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _safe_read_text(path: Path | None, max_bytes: int | None = None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        if max_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        with path.open("rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _clean_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"null", "none"}:
        return None
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value.strip("\"'")


def _extract_yaml_section_value(text: str, section: str, key: str) -> str | None:
    in_section = False
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")):
            in_section = raw_line.rstrip().startswith(f"{section}:")
            continue
        if not in_section:
            continue
        match = re.match(rf"\s+{re.escape(key)}:\s*(.+?)\s*$", raw_line)
        if match:
            return _clean_scalar(match.group(1))
    return None


def _resolve_workspace_path(raw_path: str | None) -> Path | None:
    cleaned = _clean_scalar(raw_path)
    if cleaned is None:
        return None
    path = Path(cleaned)
    if path.is_absolute():
        return path
    return BASE_DIR / path


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
    compose_containers = get_compose_container_statuses()
    megxl_runs = discover_generic_megxl_runs(compose_containers)
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
        "compose_containers": compose_containers,
        "megxl_runs": megxl_runs,
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


def _artifact_age_min(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        stat = path.stat()
        return (time.time() - max(stat.st_mtime, stat.st_ctime)) / 60
    except Exception:
        return None


def _decode_http_chunked(body: bytes) -> bytes:
    decoded = bytearray()
    pos = 0
    while pos < len(body):
        line_end = body.find(b"\r\n", pos)
        if line_end < 0:
            return body
        size_text = body[pos:line_end].split(b";", 1)[0].strip()
        try:
            size = int(size_text, 16)
        except Exception:
            return body
        pos = line_end + 2
        if size == 0:
            break
        decoded.extend(body[pos:pos + size])
        pos += size + 2
    return bytes(decoded)


def _docker_socket_request(path: str, timeout: float = 3.0) -> bytes:
    if not DOCKER_SOCKET.exists():
        return b""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(DOCKER_SOCKET))
            request = (
                f"GET {path} HTTP/1.1\r\n"
                "Host: docker\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode("ascii")
            sock.sendall(request)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except Exception:
        return b""

    raw = b"".join(chunks)
    header_end = raw.find(b"\r\n\r\n")
    if header_end < 0:
        return b""
    headers = raw[:header_end].decode("iso-8859-1", errors="replace").lower()
    body = raw[header_end + 4:]
    if "transfer-encoding: chunked" in headers:
        body = _decode_http_chunked(body)
    return body


def _docker_socket_json(path: str):
    body = _docker_socket_request(path)
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _decode_docker_log_stream(body: bytes) -> str:
    if not body:
        return ""

    # Docker's non-TTY log endpoint multiplexes stdout/stderr with 8-byte frames.
    if len(body) >= 8 and body[0] in (1, 2) and body[1:4] == b"\x00\x00\x00":
        decoded = bytearray()
        pos = 0
        while pos + 8 <= len(body):
            size = int.from_bytes(body[pos + 4:pos + 8], "big")
            pos += 8
            decoded.extend(body[pos:pos + size])
            pos += size
        body = bytes(decoded)

    return _strip_ansi(body.decode("utf-8", errors="replace"))


def _docker_socket_container_logs(container_id: str | None, lines: int = 80) -> str:
    if not container_id:
        return ""
    body = _docker_socket_request(
        f"/containers/{container_id}/logs?stdout=1&stderr=1&timestamps=0&tail={int(lines)}",
        timeout=5.0,
    )
    return _decode_docker_log_stream(body)


def _compose_statuses_from_docker_socket() -> dict[str, dict]:
    containers = _docker_socket_json("/containers/json?all=1")
    if not isinstance(containers, list):
        return {}

    statuses: dict[str, dict] = {}
    for item in containers:
        if not isinstance(item, dict):
            continue
        labels = item.get("Labels") if isinstance(item.get("Labels"), dict) else {}
        service = labels.get("com.docker.compose.service")
        if not service:
            continue

        names = item.get("Names") if isinstance(item.get("Names"), list) else []
        container_name = names[0].lstrip("/") if names else item.get("Id", "")[:12]
        container_id = item.get("Id")
        inspect = _docker_socket_json(f"/containers/{container_id}/json") if container_id else None
        state_obj = inspect.get("State", {}) if isinstance(inspect, dict) else {}
        exit_code = state_obj.get("ExitCode")

        statuses[str(service)] = {
            "service": str(service),
            "container": str(container_name),
            "container_id": str(container_id or ""),
            "state": str(item.get("State") or state_obj.get("Status") or "").lower(),
            "status_text": str(item.get("Status") or ""),
            "exit_code": str(exit_code) if exit_code is not None else None,
        }
    return statuses


def get_compose_container_statuses() -> dict[str, dict]:
    """
    Best-effort Docker Compose state. The monitor also works without Docker
    access, using filesystem artifacts only.
    """
    socket_statuses = _compose_statuses_from_docker_socket()
    if socket_statuses:
        return socket_statuses

    try:
        out = subprocess.check_output(
            ["docker", "compose", "ps", "--format", "json"],
            cwd=str(BASE_DIR),
            timeout=4,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace").strip()
    except Exception:
        return {}

    if not out:
        return {}

    rows = []
    try:
        parsed = json.loads(out)
        rows = parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        for line in out.splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    statuses: dict[str, dict] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        service = item.get("Service") or item.get("service")
        name = item.get("Name") or item.get("name")
        key = str(service or name or "").strip()
        if not key:
            continue
        state = str(item.get("State") or item.get("state") or item.get("Status") or "").lower()
        exit_code = item.get("ExitCode")
        statuses[key] = {
            "service": str(service or key),
            "container": str(name or ""),
            "state": state,
            "status_text": str(item.get("Status") or item.get("status") or ""),
            "exit_code": str(exit_code) if exit_code is not None else None,
        }
    return statuses


def _compose_declared_eval_services() -> dict[str, dict]:
    """
    Parse the local compose file just enough to map services to Hydra configs.
    This avoids hard-coding LibriBrain-specific service names.
    """
    compose_path = BASE_DIR / "docker-compose.yml"
    text = _safe_read_text(compose_path)
    if not text:
        return {}

    in_services = False
    current_name: str | None = None
    current_lines: list[str] = []
    blocks: list[tuple[str, str]] = []

    def flush_current() -> None:
        nonlocal current_name, current_lines
        if current_name is not None:
            blocks.append((current_name, "\n".join(current_lines)))
        current_name = None
        current_lines = []

    for line in text.splitlines():
        if not in_services:
            in_services = line.strip() == "services:"
            continue

        if line and not line.startswith(" "):
            flush_current()
            break

        service_match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*$", line)
        if service_match:
            flush_current()
            current_name = service_match.group(1)
            continue

        if current_name is not None:
            current_lines.append(line)

    flush_current()

    services: dict[str, dict] = {}
    for service, block in blocks:
        if service.startswith("x-"):
            continue
        if "monitor_server.py" in block:
            continue

        config_match = re.search(r"--config-name[=\s]+([A-Za-z0-9_.-]+)", block)
        module_match = re.search(r"python\s+-m\s+([A-Za-z0-9_.-]+)", block)
        if not config_match and "evaluate" not in block and "megxl" not in block.lower():
            continue

        entry = {
            "service": service,
            "config_name": config_match.group(1) if config_match else None,
            "module": module_match.group(1) if module_match else None,
        }
        services[service] = entry
    return services


def _read_config_metadata(config_name: str | None) -> dict:
    if not config_name:
        return {}
    config_path = BASE_DIR / "configs" / f"{config_name}.yaml"
    text = _safe_read_text(config_path)
    if not text:
        return {"config_path": str(config_path)}
    return {
        "config_path": str(config_path),
        "experiment_name": _extract_yaml_section_value(text, "logging", "experiment_name"),
        "save_dir": _resolve_workspace_path(_extract_yaml_section_value(text, "logging", "save_dir")),
        "checkpoint_dir": _resolve_workspace_path(_extract_yaml_section_value(text, "logging", "checkpoint_dir")),
        "dataset_type": _extract_yaml_section_value(text, "data", "dataset_type"),
        "wandb_project": _extract_yaml_section_value(text, "logging", "wandb_project"),
    }


def _merge_run(registry: dict[str, dict], name: str | None, **updates) -> None:
    if not name:
        return
    clean_name = str(name).strip()
    if not clean_name:
        return
    run = registry.setdefault(clean_name, {"exp": clean_name, "kind": "megxl_eval"})
    for key, value in updates.items():
        if value is not None and value != "":
            run[key] = value


def _find_run_name_by_save_dir(registry: dict[str, dict], save_dir: Path) -> str | None:
    return _find_run_name_by_path_key(registry, "save_dir", save_dir)


def _find_run_name_by_checkpoint_dir(registry: dict[str, dict], checkpoint_dir: Path) -> str | None:
    return _find_run_name_by_path_key(registry, "checkpoint_dir", checkpoint_dir)


def _find_run_name_by_path_key(registry: dict[str, dict], key: str, path: Path) -> str | None:
    try:
        target = path.resolve()
    except Exception:
        target = path
    for name, run in registry.items():
        candidate = run.get(key)
        if not isinstance(candidate, Path):
            continue
        try:
            if candidate.resolve() == target:
                return name
        except Exception:
            if candidate == path:
                return name
    return None


def _find_newest_log_in_dir(path: Path | None) -> Path | None:
    if path is None or not path.exists() or not path.is_dir():
        return None
    logs = sorted(path.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _iter_existing_scan_roots() -> list[Path]:
    roots = []
    for name in GENERIC_SCAN_ROOT_NAMES:
        path = BASE_DIR / name
        if path.exists():
            roots.append(path)
    return roots


def discover_generic_megxl_runs(compose_containers: dict[str, dict] | None = None) -> dict[str, dict]:
    """
    Discover MEG-XL evaluations from several generic sources:
      - services in docker-compose.yml that launch a Hydra config
      - Docker Compose runtime state
      - Hydra run directories containing .hydra/config.yaml
      - save dirs with final_results.txt / checkpoint_*.pt
    """
    registry: dict[str, dict] = {}
    compose_containers = compose_containers or {}

    declared_services = _compose_declared_eval_services()
    for service, service_meta in declared_services.items():
        cfg_meta = _read_config_metadata(service_meta.get("config_name"))
        exp_name = cfg_meta.get("experiment_name") or service
        save_dir = cfg_meta.get("save_dir")
        checkpoint_dir = cfg_meta.get("checkpoint_dir") or save_dir
        _merge_run(
            registry,
            exp_name,
            service_name=service,
            display_name=exp_name,
            config_name=service_meta.get("config_name"),
            config_path=cfg_meta.get("config_path"),
            save_dir=save_dir,
            checkpoint_dir=checkpoint_dir,
            final_results=(save_dir / "final_results.txt") if isinstance(save_dir, Path) else None,
            checkpoint_latest=(checkpoint_dir / "checkpoint_latest.pt") if isinstance(checkpoint_dir, Path) else None,
            checkpoint_best=(checkpoint_dir / "checkpoint_best.pt") if isinstance(checkpoint_dir, Path) else None,
            dataset=cfg_meta.get("dataset_type"),
            wandb_project=cfg_meta.get("wandb_project"),
            container=compose_containers.get(service),
        )

    for service, container in compose_containers.items():
        if service in declared_services:
            continue
        if not re.search(r"(eval|megxl|classification|probe)", service, re.I):
            continue
        _merge_run(
            registry,
            service,
            service_name=service,
            display_name=service,
            container=container,
        )

    for root in _iter_existing_scan_roots():
        for config_path in root.rglob(".hydra/config.yaml"):
            run_dir = config_path.parent.parent
            text = _safe_read_text(config_path)
            exp_name = (
                _extract_yaml_section_value(text, "logging", "experiment_name")
                or run_dir.name
            )
            save_dir = _resolve_workspace_path(_extract_yaml_section_value(text, "logging", "save_dir"))
            checkpoint_dir = _resolve_workspace_path(_extract_yaml_section_value(text, "logging", "checkpoint_dir")) or save_dir
            _merge_run(
                registry,
                exp_name,
                display_name=exp_name,
                hydra_dir=run_dir,
                hydra_log=_find_newest_log_in_dir(run_dir),
                config_path=config_path,
                save_dir=save_dir,
                checkpoint_dir=checkpoint_dir,
                final_results=(save_dir / "final_results.txt") if isinstance(save_dir, Path) else None,
                checkpoint_latest=(checkpoint_dir / "checkpoint_latest.pt") if isinstance(checkpoint_dir, Path) else None,
                checkpoint_best=(checkpoint_dir / "checkpoint_best.pt") if isinstance(checkpoint_dir, Path) else None,
                dataset=_extract_yaml_section_value(text, "data", "dataset_type"),
                wandb_project=_extract_yaml_section_value(text, "logging", "wandb_project"),
            )

    for root in _iter_existing_scan_roots():
        for marker in GENERIC_RUN_MARKERS[:3]:
            for path in root.rglob(marker):
                artifact_dir = path.parent
                if marker == "final_results.txt":
                    exp_name = _find_run_name_by_save_dir(registry, artifact_dir) or artifact_dir.name
                    run = registry.get(exp_name, {})
                    checkpoint_dir = run.get("checkpoint_dir") or artifact_dir
                    _merge_run(
                        registry,
                        exp_name,
                        display_name=exp_name,
                        save_dir=artifact_dir,
                        checkpoint_dir=checkpoint_dir,
                        final_results=artifact_dir / "final_results.txt",
                        checkpoint_latest=(checkpoint_dir / "checkpoint_latest.pt") if isinstance(checkpoint_dir, Path) else None,
                        checkpoint_best=(checkpoint_dir / "checkpoint_best.pt") if isinstance(checkpoint_dir, Path) else None,
                    )
                    continue

                exp_name = (
                    _find_run_name_by_checkpoint_dir(registry, artifact_dir)
                    or _find_run_name_by_save_dir(registry, artifact_dir)
                    or artifact_dir.name
                )
                run = registry.get(exp_name, {})
                save_dir = run.get("save_dir")
                checkpoint_dir = run.get("checkpoint_dir") or artifact_dir
                _merge_run(
                    registry,
                    exp_name,
                    display_name=exp_name,
                    save_dir=save_dir,
                    checkpoint_dir=checkpoint_dir,
                    final_results=(save_dir / "final_results.txt") if isinstance(save_dir, Path) else None,
                    checkpoint_latest=checkpoint_dir / "checkpoint_latest.pt",
                    checkpoint_best=checkpoint_dir / "checkpoint_best.pt",
                )

    # Drop entries without any concrete runtime, config, or artifact evidence.
    return {
        name: run
        for name, run in registry.items()
        if any(run.get(k) for k in ("service_name", "config_path", "hydra_dir", "final_results", "checkpoint_latest", "checkpoint_best"))
    }


def discover_experiments(ctx: dict | None = None) -> list[str]:
    """
    Descubre experimentos automáticamente para soportar:
      - modo clásico: task__backbone__strategy
      - modo speech-image: speech_image__<exp_id>
      - evaluaciones genéricas MEG-XL lanzadas con Docker/Hydra
    """
    plan = load_sweep_plan()
    planned_entries = _normalize_plan_experiment_entries(plan)
    planned_exps = set(planned_entries.keys())
    generic_exps = set((ctx or {}).get("megxl_runs", {}).keys())
    recent_exps = set()

    for p in LOGS_DIR.glob("*.log"):
        if p.stem.startswith("sweep_") or p.stem.startswith("precompute_"):
            continue
        age_min = _artifact_age_min(p)
        if age_min is not None and age_min < 30:
            recent_exps.add(p.stem)

    for p in CKPT_DIR.glob("*/training_state.json"):
        age_min = _artifact_age_min(p)
        if age_min is not None and age_min < 30:
            recent_exps.add(p.parent.name)

    for p in CKPT_DIR.glob("*/checkpoint_latest.pt"):
        age_min = _artifact_age_min(p)
        if age_min is not None and age_min < 30:
            recent_exps.add(p.parent.name)

    if planned_entries:
        return sorted(planned_exps | recent_exps | generic_exps)

    mode = get_sweep_mode()
    if mode == "classic":
        return sorted(set(CLASSIC_EXPS) | recent_exps | generic_exps)

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

    return sorted(exps | generic_exps)


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


def _parse_final_results_txt(path: Path | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if path is None or not path.exists():
        return metrics
    text = _safe_read_text(path)
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_./-]+)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if not match:
            continue
        try:
            metrics[match.group(1)] = float(match.group(2))
        except Exception:
            continue
    return metrics


def _pick_primary_metric(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    if not metrics:
        return None, None

    for key in ("test_f1_macro", "f1_macro", "test_balanced_acc", "balanced_acc"):
        if key in metrics:
            return key, metrics[key]

    def _retrieval_rank(item: tuple[str, float]) -> tuple[int, int, str]:
        key, _ = item
        retrieval = 0
        match = re.search(r"retrieval(\d+)", key)
        if match:
            retrieval = int(match.group(1))
        if key.startswith("balanced_") and "accuracy" in key:
            priority = 4
        elif "top" in key and "accuracy" in key:
            priority = 3
        elif "accuracy" in key:
            priority = 2
        else:
            priority = 1
        return priority, retrieval, key

    candidates = [(k, v) for k, v in metrics.items() if "loss" not in k.lower()]
    if not candidates:
        return next(iter(metrics.items()))
    candidates.sort(key=_retrieval_rank, reverse=True)
    return candidates[0]


def _read_checkpoint_summary(path: Path | None) -> dict:
    """
    Avoid importing torch in the monitor. We infer liveness from checkpoint mtimes
    and leave metric extraction to final_results.txt / logs.
    """
    if path is None or not path.exists():
        return {}
    return {"mtime": _safe_stat_mtime(path)}


def _container_status(container: dict | None) -> str | None:
    if not container:
        return None
    state = str(container.get("state") or "").lower()
    status_text = str(container.get("status_text") or "").lower()
    exit_code = container.get("exit_code")
    if "running" in state or "up" in status_text:
        return "running"
    if exit_code in {"0", 0}:
        return "done"
    if exit_code not in {None, "", "0", 0}:
        return "failed"
    if any(token in state for token in ("exited", "dead", "failed")):
        return "failed"
    return None


def _get_docker_compose_log(service: str | None, lines: int = 80) -> str:
    if not service:
        return ""
    socket_status = get_compose_container_statuses().get(service, {})
    socket_log = _docker_socket_container_logs(socket_status.get("container_id"), lines=lines)
    if socket_log.strip():
        return socket_log

    try:
        out = subprocess.check_output(
            ["docker", "compose", "logs", "--no-color", "--tail", str(lines), service],
            cwd=str(BASE_DIR),
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
        return _strip_ansi(out)
    except Exception:
        return ""


def _get_megxl_eval_status(exp: str, run: dict) -> dict:
    final_results = run.get("final_results")
    checkpoint_latest = run.get("checkpoint_latest")
    checkpoint_best = run.get("checkpoint_best")
    hydra_log = run.get("hydra_log") or _find_newest_log_in_dir(run.get("hydra_dir"))
    container = run.get("container")
    container_state = _container_status(container)

    metrics = _parse_final_results_txt(final_results)
    primary_name, primary_value = _pick_primary_metric(metrics)
    latest_age = _artifact_age_min(checkpoint_latest) if isinstance(checkpoint_latest, Path) else None
    hydra_log_age = _artifact_age_min(hydra_log) if isinstance(hydra_log, Path) else None

    status = "pending"
    if container_state == "running":
        status = "running"
    elif isinstance(final_results, Path) and final_results.exists():
        status = "done"
    elif container_state in {"done", "failed"}:
        status = container_state if metrics or container_state == "failed" else "failed"
    elif isinstance(checkpoint_latest, Path) and checkpoint_latest.exists():
        status = "running" if latest_age is not None and latest_age < 30 else "failed"
    elif isinstance(hydra_log, Path) and hydra_log.exists():
        status = "running" if hydra_log_age is not None and hydra_log_age < 30 else "failed"

    artifact_times = [
        _safe_stat_mtime(p)
        for p in (final_results, checkpoint_latest, checkpoint_best, hydra_log)
        if isinstance(p, Path)
    ]
    artifact_times = [t for t in artifact_times if t is not None]
    log_mtime = max(artifact_times) if artifact_times else None

    start_candidates = []
    for p in (run.get("hydra_dir"), checkpoint_latest, checkpoint_best, final_results, hydra_log):
        if isinstance(p, Path) and p.exists():
            try:
                start_candidates.append(p.stat().st_ctime)
            except Exception:
                pass

    elapsed_min = None
    if start_candidates:
        elapsed_min = int((time.time() - min(start_candidates)) / 60)

    last_line = ""
    if isinstance(hydra_log, Path):
        last_line = _tail_last_line(hydra_log)
    if not last_line and run.get("service_name"):
        docker_tail = _get_docker_compose_log(run.get("service_name"), lines=5)
        last_line = docker_tail.strip().splitlines()[-1][:180] if docker_tail.strip() else ""

    primary_display = primary_name.replace("_", " ") if primary_name else None

    return {
        "exp": exp,
        "display_name": run.get("display_name") or exp,
        "kind": run.get("kind", "megxl_eval"),
        "dataset": run.get("dataset"),
        "service_name": run.get("service_name"),
        "status": status,
        "epoch_current": None,
        "epoch_total": None,
        "f1_macro": primary_value,
        "balanced_acc": metrics.get("balanced_acc") or metrics.get("test_balanced_acc"),
        "best_val_f1": None,
        "primary_metric_name": primary_name,
        "primary_metric_label": primary_display,
        "primary_metric_value": primary_value,
        "metrics": metrics,
        "elapsed_min": elapsed_min,
        "last_line": last_line,
        "log_mtime": log_mtime,
    }


def get_exp_status(exp: str, ctx: dict | None = None) -> dict:
    """
    Determina el estado de un experimento a partir de los archivos en disco.
    Returns: dict con status, epoch_current, epoch_total, f1, bal_acc, elapsed_min
    """
    ctx = ctx or get_sweep_context()
    if exp in ctx.get("megxl_runs", {}):
        return _get_megxl_eval_status(exp, ctx["megxl_runs"][exp])

    plan_entry = ctx["experiments"].get(exp, {})
    log_override = ctx["log_statuses"].get("experiments", {}).get(exp, {})
    started_at = ctx["run_started_at"]

    done_sentinel  = BASE_DIR / f".exp_done_{exp}"
    job_log        = LOGS_DIR / f"{exp}.log"
    training_state = CKPT_DIR / exp / "training_state.json"
    final_results  = RESULTS_DIR / exp / "final_results.json"

    result = {
        "exp": exp,
        "display_name": exp,
        "kind": "sweep",
        "dataset": None,
        "service_name": None,
        "status": "pending",       # pending | running | done | failed | skipped
        "epoch_current": None,
        "epoch_total": None,
        "f1_macro": None,
        "balanced_acc": None,
        "best_val_f1": None,
        "primary_metric_name": "test_f1_macro",
        "primary_metric_label": "F1 macro",
        "primary_metric_value": None,
        "metrics": {},
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
    checkpoint_latest = CKPT_DIR / exp / "checkpoint_latest.pt"
    current_checkpoint = _artifact_belongs_to_current_run(checkpoint_latest, started_at)

    job_log_age_min = _artifact_age_min(job_log)
    training_state_age_min = _artifact_age_min(training_state)
    checkpoint_age_min = _artifact_age_min(checkpoint_latest)
    recent_job_log = current_job_log and job_log_age_min is not None and job_log_age_min < 10
    recent_training_state = (
        current_training_state and training_state_age_min is not None and training_state_age_min < 20
    )
    recent_checkpoint = current_checkpoint and checkpoint_age_min is not None and checkpoint_age_min < 20
    has_recent_live_artifact = recent_job_log or recent_training_state or recent_checkpoint

    if result["status"] == "skipped" and not has_recent_live_artifact:
        return result

    # ── En curso ──────────────────────────────────────────────────────────────
    # En relanzamientos, pueden quedar final_results/sentinels/logs antiguos.
    # Si hay artefactos vivos recientes, deben ganar sobre un "done" heredado.
    if has_recent_live_artifact:
        result["status"] = "running"
        if job_log_age_min is not None:
            result["log_mtime"] = job_log.stat().st_mtime
        try:
            if current_training_state:
                with open(training_state) as f:
                    ts = json.load(f)
                result["epoch_current"] = ts.get("epoch")
                result["best_val_f1"] = ts.get("metrics", {}).get("val_f1_macro")
        except Exception:
            pass
        result["last_line"] = _tail_last_line(job_log)
        if job_log.exists():
            try:
                result["elapsed_min"] = int((time.time() - job_log.stat().st_ctime) / 60)
            except Exception:
                pass
        elif training_state.exists():
            try:
                result["elapsed_min"] = int((time.time() - training_state.stat().st_ctime) / 60)
            except Exception:
                pass
        return result

    if result["status"] == "done" and not (current_results or current_done_sentinel):
        result["status"] = "pending"

    # ── Completado ────────────────────────────────────────────────────────────
    # `final_results.json` es la evidencia más fiable de finalización.
    if (result["status"] == "done" and (current_results or current_done_sentinel)) or current_results or current_done_sentinel:
        result["status"] = "done"
        try:
            with open(final_results) as f:
                d = json.load(f)
            result["f1_macro"]    = d.get("test_f1_macro")
            result["balanced_acc"] = d.get("test_balanced_acc")
            result["best_val_f1"] = d.get("best_val_f1")
            result["primary_metric_value"] = result["f1_macro"]
            result["metrics"] = d if isinstance(d, dict) else {}
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
    all_exps = discover_experiments(ctx)
    exps = [get_exp_status(e, ctx=ctx) for e in all_exps]
    precompute = get_precompute_status(ctx=ctx)
    counts = {"done": 0, "running": 0, "failed": 0, "pending": 0, "skipped": 0}
    for e in exps:
        counts[e["status"]] += 1

    completed = [
        e for e in exps
        if e["status"] == "done" and e.get("primary_metric_value") is not None
    ]
    completed.sort(key=lambda x: x.get("primary_metric_value") or 0, reverse=True)

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
        "generic_runs": len(ctx.get("megxl_runs", {})),
    }


def get_exp_log(exp: str, lines: int = 80, ctx: dict | None = None) -> str:
    """Últimas N líneas del log de un experimento del sweep activo."""
    ctx = ctx or get_sweep_context()
    generic_run = ctx.get("megxl_runs", {}).get(exp)
    if generic_run:
        service_log = _get_docker_compose_log(generic_run.get("service_name"), lines=lines)
        if service_log.strip():
            return service_log

        hydra_log = generic_run.get("hydra_log") or _find_newest_log_in_dir(generic_run.get("hydra_dir"))
        content = _read_recent_log(hydra_log, max_bytes=lines * 240, lines=lines)
        if content.strip():
            return content

        final_results = generic_run.get("final_results")
        if isinstance(final_results, Path) and final_results.exists():
            return _safe_read_text(final_results)

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
<title>MEG Monitor</title>
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
  .exp-card.done::before { background: var(--done); }
  .exp-card.running::before { background: var(--running); animation: pulse 1.5s infinite; }
  .exp-card.failed::before { background: var(--failed); }
  .exp-card.skipped::before { background: var(--skipped); }
  .exp-card.pending::before { background: var(--pending); }

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
  <div class="logo">MEG<span>·</span>MONITOR</div>
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
      <span>Progreso de ejecuciones</span>
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
        <div class="card-title" style="margin-bottom:0">Log</div>
        <select class="log-select" id="log-select" onchange="selectExp(this.value)">
          <option value="__sweep__">— sweep global —</option>
        </select>
      </div>
      <div class="log-box" id="log-box">Cargando...</div>
    </div>

    <!-- Leaderboard -->
    <div class="card">
      <div class="card-title">Leaderboard (métrica principal)</div>
      <div id="leaderboard-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Ejecución</th><th>Tipo</th>
              <th>Métrica</th><th>Valor</th><th>Extra</th>
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

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function selectExp(exp) {
  selectedExp = exp;
  document.querySelectorAll('.exp-card').forEach(c => c.classList.remove('selected'));
  const card = Array.from(document.querySelectorAll('.exp-card'))
    .find(c => c.dataset.exp === exp);
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
    const hasClassicName = parts.length >= 3;
    const displayName = e.display_name || e.exp;
    const shortName = hasClassicName ? parts.slice(1).join(' · ') : displayName;
    const taskBadge = hasClassicName
      ? (parts[0] === 'speech_image' ? 'speech' : parts[0])
      : (e.dataset || e.kind || 'megxl');
    const epochInfo = e.epoch_current
      ? `<div class="exp-epoch">Epoch ${e.epoch_current}</div>
         <div class="epoch-bar"><div class="epoch-fill" style="width:${Math.round(e.epoch_current/50*100)}%"></div></div>`
      : '';
    const metricValue = e.primary_metric_value != null ? e.primary_metric_value : e.f1_macro;
    const metricLabel = e.primary_metric_label || 'metric';
    const metric = metricValue != null
      ? `<div class="exp-metric">${esc(metricLabel)} <strong>${metricValue.toFixed(4)}</strong></div>`
      : e.best_val_f1 != null
        ? `<div class="exp-metric">Val F1 <strong>${e.best_val_f1.toFixed(4)}</strong></div>`
        : '';
    const running_icon = e.status === 'running' ? '<span class="spinner"></span>' : '';
    const sel = selectedExp === e.exp ? 'selected' : '';
    return `<div class="exp-card ${e.status} ${sel}" data-exp="${esc(e.exp)}" onclick="selectExp(this.dataset.exp)">
      <div class="exp-name">${running_icon}${esc(taskBadge)} · ${esc(shortName)}</div>
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
      opt.textContent = e.display_name || e.exp;
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
      const hasClassicName = parts.length >= 3;
      const task = hasClassicName
        ? (parts[0] === 'speech_image' ? 'speech' : parts[0])
        : (e.dataset || e.kind || 'megxl');
      const name = hasClassicName ? parts.slice(1).join(' · ') : (e.display_name || e.exp);
      const rankCls = i === 0 ? 'rank-1' : 'rank';
      const taskCls = `task-${task}`;
      const metricLabel = e.primary_metric_label || 'metric';
      const metricValue = e.primary_metric_value != null ? e.primary_metric_value : e.f1_macro;
      const extra = e.balanced_acc != null
        ? `bal ${e.balanced_acc.toFixed(4)}`
        : (e.service_name || '—');
      return `<tr>
        <td class="${rankCls}">${i === 0 ? '🥇' : i + 1}</td>
        <td>${esc(name)}</td>
        <td><span class="task-badge ${taskCls}">${esc(task)}</span></td>
        <td>${esc(metricLabel)}</td>
        <td class="metric-val">${(metricValue || 0).toFixed(4)}</td>
        <td style="color:var(--muted)">${esc(extra)}</td>
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port",     type=int, default=8080)
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    BASE_DIR    = Path(args.base_dir)
    LOGS_DIR    = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    CKPT_DIR    = BASE_DIR / "checkpoints"

    print(f"[Monitor] Leyendo datos desde: {BASE_DIR}")
    print(f"[Monitor] Dashboard disponible en: http://{args.host}:{args.port}")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Monitor] Parado.")


if __name__ == "__main__":
    main()
