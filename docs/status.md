# Experiment Status

## Active Training

| GPU | Experiment | Arch | Data | Epoch | Best Val | ETA |
|-----|-----------|------|------|-------|----------|-----|
| ? | 20K random (original) | 1024×6 | 20K | 130/200 | 1.77e-4 | ~4 hrs |
| ? | bangbang augmented | 512×4 | 10K | 20/200 | 8.08e-3 | ~6 hrs |
| ? | 20K random | 512×4 | 20K | 10/200 | 5.54e-3 | ~12 hrs |
| ? | 50K random | 1024×6 | 50K | 40/200 | 1.79e-4 | ~28 hrs |
| ? | 100K v3 | 1024×6 | 100K | 10/200 | 7.32e-4 | ~3 days |

Note: Multiple 1024×6 runs (v2-v6) share `random_20k_1024x6/`
checkpoint dir, causing overwrites. The most advanced run (ep130,
val 1.77e-4) should eventually produce the best checkpoint.
All 8 GPUs have processes with non-zero memory usage.

## Key Pending Results

1. **Bangbang augmentation** — tests velocity coverage hypothesis.
   If step/random_walk gap closes, proves the tail phenomenon
   is purely a data coverage issue. Most important pending result.
2. **20K random (1024×6)** — first complete data scaling point.
   Val loss (1.77e-4) already at oracle level; benchmark pending.
3. **20K random (512×4)** — arch × data interaction. Critical for
   determining optimal configuration.

## Completed Analyses

- Standard Benchmark v4: 23 models, see stage1d.md
- Velocity-error correlation: r=0.50 log-log
- Velocity bucket analysis: <10 rad/s = 0.5-1.7× oracle,
  15-20 rad/s = 57.7× oracle