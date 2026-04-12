# Capacity Scaling Analysis

## The Hard/Easy Split

Standard benchmark families divide into two groups:

**Easy** (multisine, chirp, sawtooth, pulse): MSE ~1.4e-4 for all methods.
**Hard** (step, random_walk): MSE 5e-3 to 5e-2, drives 95%+ of aggregate error.

| Method | Easy MSE | Hard MSE | Hard/Easy |
|--------|----------|----------|-----------|
| Active | 1.86e-4 | 5.20e-3 | **28×** |
| Random | 1.40e-4 | 7.72e-3 | 55× |
| Density | 1.54e-4 | 9.21e-3 | 60× |
| MaxEnt | 8.21e-3 | 5.53e-2 | 6.7× |

Active's advantage is entirely on hard families (1.5× better than random).
MaxEnt's low ratio is misleading — it's bad on everything, including easy families (55× worse).

## Val-Loss to Benchmark Calibration

| Method | Val-Loss | Bench MSE | Ratio |
|--------|----------|-----------|-------|
| Active | 0.00185 | 0.00186 | 1.00× |
| Random | 0.00418 | 0.00267 | 0.64× |
| MaxEnt | 0.00499 | 0.02390 | 4.79× |

**Warning**: val_loss is NOT a reliable benchmark predictor across data types.
MaxEnt's val_loss appears OK but benchmark is 4.8× worse (fails on hard families).
Random's benchmark is actually better than val_loss (easy families pull aggregate down).

## Capacity Scaling (Epoch 70 Snapshot)

All random models trained on same 10K trajectory dataset:

| Architecture | Params | best_val | vs 512×4 |
|-------------|--------|----------|----------|
| 512×4 | 1.3M | 0.00418 | baseline |
| 512×4+wd | 1.3M | 0.00629 | worse (wd hurting?) |
| 1024×4+wd | 2.6M | 0.00237 | 1.8× better |
| 2048×3 | 6.3M | 0.00274 | 1.5× better |
| **1024×6** | **4.7M** | **0.00168** | **2.5× better** |

Key insight: 1024×6 achieves val_loss BELOW active learning's benchmark (0.00186).

## Predictions

1. **hp_random_1024x6** → likely benchmark 0.001-0.002 (below active learning)
2. **DAgger 512x4** → likely best overall (task-focused data + good val_loss)
3. **Maxent models** → 10-20× worse regardless of capacity
4. **Weight decay** → mixed results, may hurt on this problem

## Implications

The active-random gap can be closed two ways:
1. **Better data** (DAgger: add benchmark-targeted trajectories)
2. **Bigger model** (1024×6: more capacity compensates for data distribution mismatch)

Both work. The optimal strategy depends on the cost of data collection vs model training.
