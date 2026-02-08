#!/bin/bash
set -e

echo "============================================"
echo "  TIVARI EXPERIMENT - TRAIN, EVAL, PR, STOP"
echo "============================================"
echo "Started: $(date)"

cd ~/hyperdata/hyperdata

# 1. Train all 4 models
echo ""
echo ">>> TRAINING"
echo ""
for cfg in training/configs/tivari_*.yaml; do
    echo "--- Training: $cfg ---"
    python training/train.py --config "$cfg"
done

# 2. Evaluate all 4 models
echo ""
echo ">>> EVALUATION"
echo ""
for variant in examples hyperdata_1pct hyperdata_5pct hyperdata_10pct; do
    model_path="checkpoints/pythia-1.4b_tivari_${variant}/final"
    if [ -d "$model_path" ]; then
        echo "--- Evaluating: $model_path ---"
        python eval/eval.py --model "$model_path" --grammar tivari --device cuda
    else
        echo "WARNING: $model_path not found, skipping"
    fi
done

# 3. Commit results and push a PR branch
echo ""
echo ">>> CREATING PR BRANCH"
echo ""
git checkout -b tivari-results-$(date +%Y%m%d) || true
git add results/ checkpoints/
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
