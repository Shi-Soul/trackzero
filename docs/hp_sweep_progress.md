# HP Sweep & DAgger Experiments - Round 2

## Motivation

After the standard benchmark ranking reversal (active #1, random #4):
1. Can larger models close the active-random gap? HP tuning never done.
2. Is maxent data fundamentally limited? Or just needs more capacity?
3. Can DAgger (task-focused augmentation) beat active learning?

## Experiment Design (8 GPUs)

| GPU | Name | Data | Arch | LR | WD | Epochs |
|-----|------|------|------|----|----|--------|
| 0 | hp_random_1024x6 | random | 1024x6 | 3e-4 | 0 | 200 |
| 1 | hp_random_2048x3 | random | 2048x3 | 3e-4 | 0 | 200 |
| 2 | hp_random_512x4_wd | random | 512x4 | 1e-4 | 1e-5 | 300 |
| 3 | hp_maxent_1024x6 | maxent | 1024x6 | 3e-4 | 0 | 200 |
| 4 | hp_maxent_2048x4 | maxent | 2048x4 | 1e-4 | 0 | 200 |
| 5 | dagger_512x4 | random+bench | 512x4 | 1e-3 | 0 | 100x5 |
| 6 | dagger_1024x4 | random+bench | 1024x4 | 3e-4 | 0 | 100x5 |
| 7 | hp_random_1024x4_wd | random | 1024x4 | 3e-4 | 1e-5 | 200 |

## Progress (Epoch ~30)

| Experiment | best_val | Status |
|-----------|---------|--------|
| hp_random_1024x6 | **0.005244** | Still improving |
| hp_random_1024x4_wd | 0.009723 | Training |
| hp_random_2048x3 | 0.010533 | Training |
| hp_random_512x4_wd | 0.012935 | Training |
| dagger_512x4 | 0.006391 | DAgger iter 0 |
| dagger_1024x4 | 0.008362 | DAgger iter 0 |
| hp_maxent_1024x6 | 0.126483 | Stuck |
| hp_maxent_2048x4 | 0.127972 | Stuck |

## Emerging Observations

1. **1024x6 random is the val_loss champion** - 0.005244, below baseline (0.010)
2. **DAgger 512x4 already at 0.0064** - excellent for a SMALLER model
3. **All maxent models stuck at ~0.127** - capacity cannot fix bad data distribution
4. **Weight decay helps random** - 1024x4+wd at 0.0097 vs baseline 0.010

## Key Insight: Density > Coverage

See density_coverage_analysis.md for the full analysis. Summary:
- Random covers 77.5% of benchmark cells with 25,940 samples/cell
- Maxent covers 100% but with only 12,764 samples/cell
- Random performs 9x better despite lower coverage
- Sample density in task-relevant cells is the key factor

---

*Status: Training ~15% complete, ~3-4 hours remaining*
