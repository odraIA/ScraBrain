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
REFRESH_SECS = 8

# Espacio de búsqueda (debe coincidir con run_sweep.sh)
TASKS      = ["phoneme", "speech"]
BACKBONES  = ["resnet18", "efficientnet_b0", "vit_tiny"]
STRATEGIES = ["frozen", "partial_ft"]
ALL_EXPS   = [
    f"{t}__{b}__{s}"
    for t in TASKS
    for b in BACKBONES
    for s in STRATEGIES
]


# ==============================================================================
# LÓGICA DE ESTADO
# ==============================================================================

def get_exp_status(exp: str) -> dict:
    """
    Determina el estado de un experimento a partir de los archivos en disco.
    Returns: dict con status, epoch_current, epoch_total, f1, bal_acc, elapsed_min
    """
    done_sentinel  = BASE_DIR / f".exp_done_{exp}"
    job_log        = LOGS_DIR / f"{exp}.log"
    training_state = CKPT_DIR / exp / "training_state.json"
    final_results  = RESULTS_DIR / exp / "final_results.json"

    result = {
        "exp": exp,
        "status": "pending",       # pending | running | done | failed
        "epoch_current": None,
        "epoch_total": None,
        "f1_macro": None,
        "balanced_acc": None,
        "best_val_f1": None,
        "elapsed_min": None,
        "last_line": "",
    }

    # ── Completado ────────────────────────────────────────────────────────────
    if done_sentinel.exists():
        result["status"] = "done"
        if final_results.exists():
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
    if job_log.exists():
        try:
            mtime = job_log.stat().st_mtime
            age_min = (time.time() - mtime) / 60
            # Si el log se modificó hace menos de 10 min, está corriendo
            if age_min < 10:
                result["status"] = "running"
            else:
                # Log existe pero inactivo → probablemente falló
                result["status"] = "failed"

            # Leer training_state.json para epoch actual
            if training_state.exists():
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
    exps = [get_exp_status(e) for e in ALL_EXPS]
    counts = {"done": 0, "running": 0, "failed": 0, "pending": 0}
    for e in exps:
        counts[e["status"]] += 1

    completed = [e for e in exps if e["status"] == "done" and e["f1_macro"] is not None]
    completed.sort(key=lambda x: x["f1_macro"] or 0, reverse=True)

    # Leer sweep log global
    sweep_logs = sorted(LOGS_DIR.glob("sweep_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    sweep_tail = ""
    if sweep_logs:
        try:
            with open(sweep_logs[0], "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 2000))
                sweep_tail = f.read().decode("utf-8", errors="replace")
        except Exception:
            pass

    # GPU stats via nvidia-smi
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

    return {
        "timestamp": datetime.now().isoformat(),
        "total": len(ALL_EXPS),
        "counts": counts,
        "experiments": exps,
        "leaderboard": completed,
        "sweep_tail": sweep_tail,
        "gpu_info": gpu_info,
    }


def get_exp_log(exp: str, lines: int = 80) -> str:
    """Últimas N líneas del log de un experimento."""
    log_path = LOGS_DIR / f"{exp}.log"
    if not log_path.exists():
        return f"[Sin log todavía para {exp}]"
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - lines * 200))
            content = f.read().decode("utf-8", errors="replace")
        tail = "\n".join(content.split("\n")[-lines:])
        return tail
    except Exception as e:
        return f"[Error leyendo log: {e}]"


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

function render(d) {
  // Counters
  document.getElementById('cnt-done').textContent    = d.counts.done;
  document.getElementById('cnt-running').textContent = d.counts.running;
  document.getElementById('cnt-failed').textContent  = d.counts.failed;
  document.getElementById('cnt-pending').textContent = d.counts.pending;

  // Progress
  const pct = Math.round(d.counts.done / d.total * 100);
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = `${d.counts.done} / ${d.total}`;

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
    const taskBadge = parts[0];
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
    if (!prevOptions.has(e.exp) && e.status !== 'pending') {
      const opt = document.createElement('option');
      opt.value = e.exp;
      opt.textContent = e.exp;
      select.appendChild(opt);
    }
  });

  // Leaderboard
  const tbody = document.getElementById('leaderboard-body');
  if (d.leaderboard.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);text-align:center;padding:24px">Sin resultados todavía...</td></tr>';
  } else {
    tbody.innerHTML = d.leaderboard.map((e, i) => {
      const parts = e.exp.split('__');
      const task = parts[0];
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
            if exp == "__sweep__":
                sweep_logs = sorted(
                    LOGS_DIR.glob("sweep_*.log"),
                    key=lambda p: p.stat().st_mtime, reverse=True
                )
                if sweep_logs:
                    text = get_exp_log.__func__(None, None) if False else ""
                    try:
                        with open(sweep_logs[0], "rb") as f:
                            f.seek(0, 2)
                            f.seek(max(0, f.tell() - lines * 200))
                            text = f.read().decode("utf-8", errors="replace")
                        text = "\n".join(text.split("\n")[-lines:])
                    except Exception as e:
                        text = str(e)
                else:
                    text = "[Sin sweep log todavía]"
                self.send_text(text)
            else:
                self.send_text(get_exp_log(exp, lines))

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
