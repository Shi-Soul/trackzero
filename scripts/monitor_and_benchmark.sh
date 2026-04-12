#!/bin/bash
# Monitor training progress and run benchmark when models are ready
cd /NAS2020/Workspaces/DRLGroup/wjxie/robotics/trackzero

MODELS=(
  "hp_random_1024x6_lr3e4"
  "hp_random_2048x3_lr3e4" 
  "hp_random_512x4_lr1e4_wd"
  "hp_maxent_1024x6_lr3e4"
  "hp_maxent_2048x4_lr1e4"
  "hp_random_1024x4_lr3e4_wd"
  "dagger_benchmark_512x4"
  "dagger_benchmark_1024x4"
)

while true; do
    echo "=== CHECK $(date) ==="
    DONE=0
    TOTAL=${#MODELS[@]}
    
    for m in "${MODELS[@]}"; do
        if [ -f "outputs/$m/best_model.pt" ]; then
            echo "  DONE: $m"
            DONE=$((DONE + 1))
        else
            # Show latest log line
            LOG="outputs/${m}.log"
            if [ -f "$LOG" ]; then
                echo "  TRAINING: $m — $(tail -1 $LOG)"
            else
                echo "  TRAINING: $m — no log"
            fi
        fi
    done
    
    echo "  Progress: $DONE / $TOTAL complete"
    
    if [ $DONE -eq $TOTAL ]; then
        echo "ALL DONE! Running standard benchmark..."
        TMPDIR=/NAS2020/Workspaces/DRLGroup/wjxie/tmp_warp \
        WARP_CACHE_PATH=/NAS2020/Workspaces/DRLGroup/wjxie/warp_cache \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONUNBUFFERED=1 \
        UV_LINK_MODE=copy uv run python scripts/standard_benchmark.py \
          --device cuda:0 \
          --output outputs/standard_benchmark_v2.json 2>&1
        echo "BENCHMARK COMPLETE!"
        break
    fi
    
    sleep 300  # Check every 5 minutes
done
