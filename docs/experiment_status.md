# Running Experiments Status

*Last updated: 2025-04-12 18:20 UTC*

## Active Experiments (8 GPUs)

| GPU | Name | Arch | Data | Progress | Best Val | Notes |
|-----|------|------|------|----------|----------|-------|
| 0 | hp_random_1024x6 | 1024×6 | 10K random | ep 90/200 | 0.001675 | Best HP sweep |
| 1 | hp_random_2048x3 | 2048×3 | 10K random | ep 90/200 | 0.002131 | Worse than 1024×6 despite more params |
| 2 | hp_random_512x4_wd | 512×4 | 10K random | ep 100/300 | 0.004825 | Weight decay, slow convergence |
| 3 | hp_maxent_1024x6 | 1024×6 | 10K maxent | ep 90/200 | 0.021425 | Maxent data still bad |
| 4 | dagger_1024x4 | 1024×4 | 11.2K DAgger | ep 1/100 | 0.085 | Just restarted (GPU 4 has zombie) |
| 5 | dagger_512x4 | 512×4 | 11.2K DAgger | ep 80/100 | **0.001202** | ★ BEST OVERALL |
| 6 | 20K_random | 1024×6 | 20K random | ep 10/200 | 0.001741 | Data scaling test |
| 7 | hp_random_1024x4_wd | 1024×4 | 10K random | ep 90/200 | 0.002131 | Matches 2048×3 |

## Key Observations

1. **DAgger 512×4** (0.001202) is the overall val-loss champion
2. **20K random** (0.001741 at ep10!) is on track to beat HP sweep models
3. **Architecture**: 1024×6 > 1024×4 ≈ 2048×3 > 512×4 (depth > width)
4. **Maxent data**: 1024×6 can't save bad data distribution (21× worse than random)
5. **Weight decay**: minor benefit (0.002131 vs 0.002301 baseline)

## Pending Results

- DAgger 512×4 iter 0 benchmark (ETA ~30 min)
- DAgger 1024×4 full training (ETA ~3h)
- HP sweep models completion (ETA ~2h)
- 20K random full training (ETA ~3h)

## Completed Benchmarks

See `outputs/standard_benchmark.json` (11 original models)
and `outputs/probe_benchmark.json` (18 probe models).
