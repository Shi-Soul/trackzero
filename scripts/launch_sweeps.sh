#!/bin/bash
# Launch HP sweeps across all 8 GPUs in parallel.
# Each GPU sweeps a different training data source.
#
# Usage: bash scripts/launch_sweeps.sh

set -e
cd "$(dirname "$0")/.."

export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
export TMPDIR=/NAS2020/Workspaces/DRLGroup/wjxie/tmp_warp
export WARP_CACHE_PATH=/NAS2020/Workspaces/DRLGroup/wjxie/warp_cache

LOG_DIR=outputs/sweep_logs
mkdir -p $LOG_DIR

echo "Launching HP sweeps on 8 GPUs..."

# GPU 0: maxent_rl (the key one — does more capacity help?)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/hp_sweep.py \
  --data-dir outputs/stage1c_maxent_rl \
  --tag maxent_rl --gpu 0 \
  > $LOG_DIR/maxent_rl.log 2>&1 &
echo "  GPU 0: maxent_rl sweep"

# GPU 1: random_matched baseline (confirm best HP)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/hp_sweep.py \
  --data-h5 data/medium/train.h5 \
  --tag random_matched --gpu 0 \
  > $LOG_DIR/random_matched.log 2>&1 &
echo "  GPU 1: random_matched sweep"

# GPU 2: supervised (1a_scaled)
CUDA_VISIBLE_DEVICES=2 uv run python scripts/hp_sweep.py \
  --data-h5 data/medium/train.h5 \
  --tag supervised --gpu 0 \
  > $LOG_DIR/supervised.log 2>&1 &
echo "  GPU 2: supervised sweep"

# GPU 3: hybrid
CUDA_VISIBLE_DEVICES=3 uv run python scripts/hp_sweep.py \
  --data-dir outputs/stage1c_hybrid_full \
  --tag hybrid --gpu 0 \
  > $LOG_DIR/hybrid.log 2>&1 &
echo "  GPU 3: hybrid sweep"

# GPU 4: density
CUDA_VISIBLE_DEVICES=4 uv run python scripts/hp_sweep.py \
  --data-dir outputs/stage1c_density_full \
  --tag density --gpu 0 \
  > $LOG_DIR/density.log 2>&1 &
echo "  GPU 4: density sweep"

# GPU 5: adversarial
CUDA_VISIBLE_DEVICES=5 uv run python scripts/hp_sweep.py \
  --data-dir outputs/stage1c_adversarial_full \
  --tag adversarial --gpu 0 \
  > $LOG_DIR/adversarial.log 2>&1 &
echo "  GPU 5: adversarial sweep"

# GPU 6: active (disagreement)
CUDA_VISIBLE_DEVICES=6 uv run python scripts/hp_sweep.py \
  --data-dir outputs/stage1c_active_full \
  --tag active --gpu 0 \
  > $LOG_DIR/active.log 2>&1 &
echo "  GPU 6: active sweep"

# GPU 7: rebalance
CUDA_VISIBLE_DEVICES=7 uv run python scripts/hp_sweep.py \
  --data-dir outputs/stage1c_rebalance_full \
  --tag rebalance --gpu 0 \
  > $LOG_DIR/rebalance.log 2>&1 &
echo "  GPU 7: rebalance sweep"

echo ""
echo "All sweeps launched. Logs in $LOG_DIR/"
echo "Monitor with: tail -f $LOG_DIR/*.log"
echo "Check GPU usage: nvidia-smi"

wait
echo "All sweeps complete!"
