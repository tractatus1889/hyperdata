#!/bin/bash
set -e

echo "============================================"
echo "  TIVARI EXPERIMENT - TRAIN, EVAL, PR, STOP"
echo "============================================"
echo "Started: $(date)"

cd ~/hyperdata/hyperdata

# 1. Train and evaluate all 4 models
python run_experiment.py --grammar tivari

# 3. Commit results and push a PR branch
echo ""
echo ">>> CREATING PR BRANCH"
echo ""
git checkout -b tivari-results-$(date +%Y%m%d) || true
git add results/
git commit -m "Add tivari experiment results

Trained and evaluated 4 tivari models:
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

# 4. Shut down the instance
echo ""
echo ">>> SHUTTING DOWN IN 60 SECONDS (Ctrl+C to cancel)"
sleep 60
sudo shutdown -h now
