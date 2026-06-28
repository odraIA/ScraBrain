launch_worker "${GPUS[0]}" 0
if [[ ${#GPUS[@]} -gt 1 ]]; then
  launch_worker "${GPUS[1]}" 1
fi
if [[ ${#GPUS[@]} -gt 2 ]]; then
  launch_worker "${GPUS[2]}" 2
fi
printf 'Sweep launched: %s\n' "$RUN_ROOT"
printf 'Queue: %s\n' "$QUEUE_FILE"
printf 'Run table: %s\n' "$RUNS_FILE"
printf 'Stage 2 starts only after the matching reading checkpoint succeeds.\n'
