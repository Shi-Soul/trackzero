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

## Progress (Epoch ~60)

| Experiment | Epoch | best_val | Trend |
|-----------|-------|---------|-------|
| hp_random_1024x6 | 60/200 | **0.002333** | Excellent, still improving |
| hp_random_1024x4_wd | 60/200 | **0.002630** | Strong |
| hp_random_2048x3 | 60/200 | **0.002790** | Good |
| hp_random_512x4_wd | 70/300 | 0.006290 | Improving slowly |
| dagger_512x4 | 60/100 | **0.001540** | ★ BEST of all ★ |
| dagger_1024x4 | 60/100 | **0.001681** | ★ Excellent ★ |
| hp_maxent_1024x6 | 60/200 | 0.034639 | Improving but 15× worse |
| hp_maxent_2048x4 | 50/200 | 0.095969 | Slowly improving, still terrible |

## Emerging Observations

1. **DAgger 512x4 is the overall val_loss champion** at 0.001540
2. **DAgger 1024x4 close behind** at 0.001681
3. **HP-tuned random is very strong** - 1024x6 at 0.002333 (near active's benchmark 0.00186)
4. **Maxent still terrible** - even 2048x4 (12.6M params) stuck at 0.096
5. **Weight decay helps** - 512x4+wd improved from 0.010 to 0.006 baseline

## Predictions for Standard Benchmark

Based on val_loss trends and the density > coverage principle:
- DAgger models should outperform active learning (best_val 0.0015 < active's benchmark 0.0019)
- HP random 1024x6 will be competitive with active (0.0023 val_loss)
- Maxent will remain 10-15× worse regardless of model size
