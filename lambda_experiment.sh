#!/bin/bash
set -e

GRAMMAR=${1:?Usage: ./lambda_experiment.sh <grammar> [checkpoint]}
CHECKPOINT=${2:-}

echo "============================================"
echo "  EXPERIMENT: ${GRAMMAR} - TRAIN, EVAL, PR, STOP"
echo "============================================"
echo "Grammar: ${GRAMMAR}"
echo "Checkpoint: ${CHECKPOINT:-latest}"
echo "Started: $(date)"

cd ~/hyperdata/hyperdata

# 1. Train and evaluate all 4 models
EXPERIMENT_CMD="python run_experiment.py --grammar ${GRAMMAR}"
if [ -n "${CHECKPOINT}" ]; then
    EXPERIMENT_CMD="${EXPERIMENT_CMD} --checkpoint ${CHECKPOINT}"
fi
${EXPERIMENT_CMD}

echo ""
echo "============================================"
echo "  DONE!"
echo "============================================"
echo "Finished: $(date)"

# 2. Shut down the instance
echo ""
echo ">>> SHUTTING DOWN IN 60 SECONDS (Ctrl+C to cancel)"
sleep 60
sudo shutdown -h now
