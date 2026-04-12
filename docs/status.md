# Current Experiment Status

> Last updated: session checkpoint (see git log for date)

## Running Experiments

| GPU | Experiment | Progress | Notes |
|-----|-----------|----------|-------|
| 0 | hp_random_1024x6 | ep170/200 | val=1.42e-3 |
| 1 | 50K random v2 | ep1/200 | 1024×6, data gen complete |
| 4 | (zombie) | — | 10.4GB, container PID issue |
| 5 | dagger_1024x6 | ep1/100 | iter 0 |
| 6 | 20K random | CRASHED | NFS .nfs busy error |

## Completed Experiments

- Standard benchmark v2: 13 models evaluated → `outputs/standard_benchmark_v2.json`
- HP sweep 1024×4: complete
- HP sweep 512×4 variants: complete
- All Stage 1C variants: complete
- DAgger 512×4 v1/v2: complete (catastrophic)
- DAgger 1024×4: iter3 ep60

## Key Pending Results

1. 20K random needs restart (crashed on NFS cleanup)
2. 50K random training in progress
3. HP 1024×6 nearing completion (ep170/200)
4. DAgger 1024×6 just started

## Benchmark Queue

When training completes, run `scripts/standard_benchmark.py` to evaluate:
- hp_random_1024x6 (from HP sweep)
- random_20k (when restarted and complete)
- random_50k (when complete)
- dagger iterations (after each iter)
