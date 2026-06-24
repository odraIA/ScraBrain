cat > "$PIPE_DIR/metadata.txt" <<META
Pipeline: $INIT
Order: reading -> listening
Reading datasets: EEGDash (delong, control) + ZuCo (NR)
Reading experiment: $READ_EXP
Listening experiment: $LISTEN_EXP
META
LAST_CKPT=""
run_stage reading train_criss_cross_eeg_reading_continuous "$READ_EXP" "$READ_BATCH"
READ_CKPT="$LAST_CKPT"
printf '%s\n' "$READ_CKPT" > "$PIPE_DIR/reading_checkpoint_used_for_listening.txt"
run_stage listening train_criss_cross_eeg_listening_continuous "$LISTEN_EXP" "$LISTEN_BATCH" "$READ_CKPT"
cat > "$PIPE_DIR/final_results.txt" <<RESULT
Completed: $INIT
Reading datasets: EEGDash + ZuCo
Reading checkpoint passed to listening: $READ_CKPT
Final listening checkpoint: $LAST_CKPT
RESULT
