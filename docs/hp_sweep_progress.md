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

## Progress (Epoch ~70)

| Experiment | Epoch | best_val | Trend |
|-----------|-------|---------|-------|
| **hp_random_1024x6** | 70/200 | **0.001675** | ★ Below active's benchmark! ★ |
| hp_random_1024x4_wd | 70/200 | **0.002366** | Strong, still improving |
| hp_random_2048x3 | 60/200 | **0.002790** | Good |
| hp_random_512x4_wd | 70/300 | 0.006290 | Improving slowly |
| dagger_512x4 | 60/100 | **0.001540** | ★ BEST val_loss ★ |
| dagger_1024x4 | 60/100 | **0.001681** | ★ Excellent ★ |
| hp_maxent_1024x6 | 60/200 | 0.034639 | 15× worse, hopeless |
| hp_maxent_2048x4 | ~10/200 | >0.095 | Very slow, 12.6M params |

## Key Finding: Capacity Can Close the Gap

**hp_random_1024x6 at epoch 70: val_loss = 0.001675 — BELOW active learning's benchmark MSE (0.001858)**

This means with sufficient model capacity (1024×6 = 4.7M params vs 512×4 = 1.3M):
- Random data + large model ≈ active learning + default model
- DAgger still leads (0.001540) — benchmark-focused data is most effective

## Error Concentration Analysis (Active Learning Model)

Fixing just 10 worst step trajectories → 58.5% aggregate improvement.
Fixing 10 worst step + 10 worst random_walk → 78.9% improvement.
Theoretical floor (all trajectories at median): 1.37e-4 (92.6% below current).

This explains why DAgger works: it adds data targeting exactly these hard trajectories.

## Predictions for Standard Benchmark

1. DAgger models → likely #1 (val_loss 0.0015 and benchmark-focused training)
2. hp_random_1024x6 → likely #2, competitive with active (val_loss 0.0017)
3. Active learning → #3 with current 1.86e-3
4. Maxent → still 10-15× worse regardless of capacity
