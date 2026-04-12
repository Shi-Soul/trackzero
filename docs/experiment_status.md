# Running Experiments Status

*Last updated: 2025-04-12 20:30 UTC*

## Active Experiments

| GPU | Name | Progress | Best Val | Status |
|-----|------|----------|----------|--------|
| 0 | hp_random_1024x6 | ep160/200 | 0.001426 | ✅ Alive (~40 ep left) |
| 1 | **50K_random v2** | data gen | — | ✅ Just launched |
| 2 | hp_random_2048x3 | ep150/200 | 0.001632 | ⚠️ ~10min stale |
| 3 | dagger_1024x4 | ep40/100 | 0.001426 | ⚠️ ~7min stale |
| 4 | (zombie processes) | — | — | ❌ Unusable |
| 5 | **dagger_1024x6** | ep1/100 | 0.084 | ✅ Just launched |
| 6 | **20K_random** | ep110/200 | **0.000229** | ✅ ALIVE (★ star exp) |
| 7 | dagger_512x4 v3 | ep1/100 | 0.084 | ⚠️ Just launched |

## Key Pending Results

1. **20K random benchmark**: needs checkpoint (saves at end, ep200)
2. **50K random**: data gen started, will train 200 epochs on 1024×6
3. **HP sweep checkpoints**: should save when 1024x6 and 2048x3 finish
4. **DAgger full iteration curve**: need iter 1-4 (current: iter 0)

## Completed Benchmark Results

See `outputs/standard_benchmark_v2.json` — 13 models evaluated.
Oracle: `outputs/oracle_benchmark.json`. Probes: `outputs/probe_benchmark.json`.
