docker compose -f "$TRAIN_YML" build eeg_train_reading_listening
launch "${GPUS[0]}" pretrained
launch "${GPUS[1]}" from_scratch
echo "Sweep launched: $RUN_ROOT"
echo "Both containers execute reading first and listening only after reading succeeds."
