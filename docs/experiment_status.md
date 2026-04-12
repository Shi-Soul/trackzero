# Running Experiments Status

*Last updated: 2025-04-12 19:10 UTC*

## Active Experiments

| GPU | Name | Arch | Data | Progress | Best Val | Notes |
|-----|------|------|------|----------|----------|-------|
| 0 | hp_random_1024x6 | 1024×6 | 10K random | ep120/200 | 0.001445 | Plateauing |
| 1 | hp_random_2048x3 | 2048×3 | 10K random | ep110/200 | 0.001859 | Still improving |
| 2 | **50K_random** | 1024×6 | **50K random** | data gen | — | ★ KEY scaling test |
| 3 | dagger_1024x4 | 1024×4 | 11.2K DAgger | ep20/100 | 0.005 | Iter 0 training |
| 4 | (zombie) | — | — | — | — | Unusable |
| 5 | dagger_512x4_v2 | 512×4 | 11.2K DAgger | iter0 ep1 | 0.081 | Fresh restart |
| 6 | **20K_random** | 1024×6 | 20K random | **ep50/200** | **0.000467** | ★ BEST |
| 7 | hp_random_1024x4_wd | 1024×4 | 10K random | ep120/200 | 0.001967 | Plateauing |

## 20K Random — Star Experiment

Learning curve (the most important data in this session):

| Epoch | best_val | vs Oracle | vs Active benchmark |
|-------|----------|-----------|---------------------|
| 10 | 1.741e-3 | 9.4× | 0.94× |
| 20 | 1.169e-3 | 6.3× | 0.63× |
| 30 | 7.90e-4 | 4.3× | 0.42× |
| 40 | 4.81e-4 | 2.6× | 0.26× |
| 50 | 4.67e-4 | 2.5× | 0.25× |

Curve flattening at ep40-50 (3% improvement vs 39% at ep30-40).
Final prediction: ~3-4e-4 (val-loss), but benchmark MSE TBD.

## DAgger Results

DAgger 512×4 iter 0: val=1.14e-3, **benchmark MSE=0.639** (terrible!)
This confirms: val-loss ≠ benchmark. Compounding errors in closed-loop.
Iterations 1+ should improve via on-policy data correction.

## Completed Benchmarks

See `outputs/standard_benchmark.json` (11 original models)
and `outputs/probe_benchmark.json` (18 probe models).
