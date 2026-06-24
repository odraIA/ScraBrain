completed() {
  local f="${LOG_ROOT}/$1/final_results.json"
  [[ -s "$f" ]] && python3 -c 'import json,sys;sys.exit(json.load(open(sys.argv[1])).get("status")!="completed")' "$f"
}

best_ckpt() {
  local d="${CKPT_ROOT}/$1" f
  [[ -d "$d" ]] || return 1
  for f in "$d/checkpoint_best.pt" "$d/checkpoint_latest.pt" "$d/last.ckpt"; do
    [[ -s "$f" ]] && { echo "$f"; return 0; }
  done
  f="$(find "$d" -maxdepth 1 -type f -name 'checkpoint-*.ckpt' -printf '%f\n' 2>/dev/null | sort -V | tail -1)"
  [[ -n "$f" ]] && echo "$d/$f"
}

resume_ckpt() {
  local d="${CKPT_ROOT}/$1" f
  [[ -d "$d" ]] || return 1
  for f in "$d/last.ckpt" "$d/checkpoint_latest.pt"; do
    [[ -s "$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}
