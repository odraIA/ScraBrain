CORRECCIÓN DEL PIPELINE FULL-BAND EEG

Este paquete sustituye el launcher listening-only por dos pipelines completos:

1. MEG-XL -> reading (EEGDash + ZuCo) -> listening
2. scratch -> reading (EEGDash + ZuCo) -> listening

La etapa listening solo comienza cuando reading termina correctamente y recibe
explícitamente el checkpoint producido por su propia etapa reading.

Instalación:
  bash install.sh ~/proyectos/meegxl/ScraBrain

Lanzamiento:
  cd ~/proyectos/meegxl/ScraBrain
  mkdir -p logs
  RUN_LOG="logs/full_band_reading_then_listening_$(date +%Y%m%d_%H%M%S).log"
  nohup env EEG_GPUS="0 1" EEG_BATCH_SIZE=4     bash scripts/run_eeg_full_band_reading_then_listening_sweep.sh     > "$RUN_LOG" 2>&1 < /dev/null &
  echo "$RUN_LOG"

El launcher prepara primero la caché de reading, después la de listening, y lanza
dos contenedores Docker detached. Cada contenedor contiene el pipeline completo,
por lo que cerrar la sesión SSH no interrumpe la transición reading -> listening.

Los checkpoints antiguos de listening-only no se reutilizan.
