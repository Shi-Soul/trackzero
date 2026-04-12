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

### Early Validation

20K random training (1024×6 architecture):
- Epoch 10: best_val = **0.001741**
- Epoch 20: best_val = **0.001169** ← already beating ALL other models!

Estimated final (with 20-30% further improvement): **~9.4e-4**
This is **2× better than active learning** (1.86e-3) with pure random data.

### Updated Predictions

| N | Predicted Final | Gap to Oracle |
|---|----------------|---------------|
| 20K | ~9.4e-4 | 5.1× |
| 50K | ~4.3e-4 | 2.3× |
| 100K | ~2.4e-4 | 1.3× |
| 135K | ~1.9e-4 | ~1.0× (oracle) |

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
