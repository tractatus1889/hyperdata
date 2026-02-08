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

# 2. Commit results and push a PR branch
echo ""
echo ">>> CREATING PR BRANCH"
echo ""
BRANCH_NAME="${GRAMMAR}-results-$(date +%Y%m%d)"
if [ -n "${CHECKPOINT}" ]; then
    BRANCH_NAME="${GRAMMAR}-${CHECKPOINT}-results-$(date +%Y%m%d)"
fi
git checkout -b "${BRANCH_NAME}" || true
git add results/
git commit -m "Add ${GRAMMAR} experiment results

Trained and evaluated 4 ${GRAMMAR} models:
- examples only
- hyperdata 1%
- hyperdata 5%
- hyperdata 10%

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" || echo "Nothing to commit"
git push -u origin HEAD

BRANCH=$(git branch --show-current)
echo ""
echo "============================================"
echo "  DONE! Create PR at:"
echo "  https://github.com/tractatus1889/hyperdata/compare/${BRANCH}?expand=1"
echo "============================================"
echo "Finished: $(date)"

# 3. Shut down the instance
echo ""
echo ">>> SHUTTING DOWN IN 60 SECONDS (Ctrl+C to cancel)"
sleep 60
sudo shutdown -h now
