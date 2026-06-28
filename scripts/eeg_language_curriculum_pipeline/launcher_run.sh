nohup python3 scripts/eeg_dynamic_gpu_scheduler.py \
  --compose-file "$TRAIN_YML" \
  --run-root "$RUN_ROOT" \
  --stamp "$STAMP" \
  --gpus "${GPUS[@]}" \
  > "$RUN_ROOT/gpu_scheduler.log" 2>&1 < /dev/null &

SCHEDULER_PID=$!
printf '%s\n' "$SCHEDULER_PID" > "$RUN_ROOT/gpu_scheduler.pid"

printf 'Sweep scheduler launched: %s\n' "$RUN_ROOT"
printf 'Scheduler PID: %s\n' "$SCHEDULER_PID"
printf 'GPU pool: %s\n' "${GPUS[*]}"
printf 'Scheduler log: %s\n' "$RUN_ROOT/gpu_scheduler.log"
printf 'Queue: %s\n' "$QUEUE_FILE"
printf 'Run table: %s\n' "$RUNS_FILE"
printf 'Each pipeline waits independently for the first free GPU.\n'
printf 'Stage 2 starts only after the matching reading checkpoint succeeds.\n'
