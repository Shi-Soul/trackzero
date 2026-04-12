# Scaling Laws and Predictions

## Data Scaling Law

From ablation experiments (1K, 2K, 5K, 10K random trajectories):

```
MSE ∝ N^{-0.85}
```

where N = number of training trajectories.

**Doubling data → 0.56× MSE** (45% reduction per doubling).

### Predictions (with 1024×6 architecture, 200 epochs)

| N (trajectories) | Predicted MSE | vs Active (1.86e-3) |
|------------------|--------------|---------------------|
| 10K | 2.67e-3 | 1.44× worse |
| 20K | 1.48e-3 | **0.80×** (beats active) |
| 50K | 6.8e-4 | 0.37× |
| 100K | 3.8e-4 | 0.20× |
| 350K | 1.3e-4 | 0.07× (≈oracle) |

### Validated: 20K Random Training (1024×6)

| Epoch | best_val | vs Oracle (1.85e-4) |
|-------|----------|---------------------|
| 10 | 1.741e-3 | 9.4× |
| 20 | 1.169e-3 | 6.3× |
| 30 | 7.90e-4 | 4.3× |
| 40 | **4.81e-4** | **2.6×** |

**Training dynamics**: val ~ epoch^{-0.887} (R²=0.94)

Predictions for 20K model:
| Epoch | Predicted val | Gap to Oracle |
|-------|--------------|---------------|
| 50 | 4.55e-4 | 2.5× |
| 100 | 2.46e-4 | 1.3× |
| 150 | 1.72e-4 | 0.93× (!) |
| 200 | 1.33e-4 | 0.72× (!) |

**Model may surpass FD oracle** in val-loss by epoch ~138.
(But val-loss ≠ benchmark MSE — awaiting final benchmark eval.)

### 10K vs 20K Direct Comparison (same 1024×6 arch)

| Epoch | 10K val | 20K val | Ratio |
|-------|---------|---------|-------|
| 10 | 1.64e-2 | 1.74e-3 | 9.4× |
| 20 | 6.79e-3 | 1.17e-3 | 5.8× |
| 30 | 5.24e-3 | 7.90e-4 | 6.6× |
| 40 | 3.11e-3 | 4.81e-4 | 6.5× |

2× data → 6.5× better val-loss at same epoch. Data quantity dominates.

## Architecture Scaling

| Architecture | Params | Val Loss (ep 90) | Notes |
|-------------|--------|------------------|-------|
| 512×4 | 794K | 0.004825 | Baseline |
| 1024×4 | 2.6M | 0.002131 | 3.3× params → 2.3× better |
| 1024×6 | 5.3M | 0.001675 | 6.7× params → 2.9× better |
| 2048×3 | 8.4M | 0.002131 | 10.6× params → 2.3× better |

**Key finding**: depth matters more than width.
1024×6 (5.3M params) beats 2048×3 (8.4M params) with 37% fewer parameters.

## Combined Scaling Hypothesis

The two scaling axes are approximately independent:
- Data scaling: MSE ~ N^{-0.85}
- Architecture: ~2× improvement from 512×4 → 1024×6

Best predicted combination: 50K trajectories + 1024×6 architecture
→ predicted MSE ≈ **3.4e-4** (1.8× oracle).

## Comparison with Active Learning

Active learning achieves 1.86e-3 with only 10K trajectories
and 512×4 architecture. To match active with random data:
- Same architecture (512×4): need ~25K trajectories
- With 1024×6: need ~15K trajectories

Active learning's advantage is equivalent to ~2.5× data efficiency.
This is significant but NOT magical — brute-force data scaling
can overcome it.
