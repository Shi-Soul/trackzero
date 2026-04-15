#!/bin/bash
# Capacity scaling: factored architecture at 10-DOF with different model sizes
# Tests whether the 10-DOF error floor (0.196) is capacity-limited
cd /NAS2020/Workspaces/DRLGroup/wjxie/robotics/trackzero

echo "===== CAPACITY SCALING: 10-link factored ====="
echo "Testing: 256x4, 1024x4, 1024x6"
echo ""

for H in 256 1024; do
  L=4
  echo "============================================"
  echo "Config: factored ${H}x${L}, 10-link"
  echo "============================================"
  .venv/bin/python -u -m scripts.run_structured --n-links 10 --hidden $H --layers $L
  echo ""
done

# Also test 1024x6
echo "============================================"
echo "Config: factored 1024x6, 10-link"
echo "============================================"
.venv/bin/python -u -m scripts.run_structured --n-links 10 --hidden 1024 --layers 6
echo ""

echo "===== CAPACITY SCALING COMPLETE ====="
