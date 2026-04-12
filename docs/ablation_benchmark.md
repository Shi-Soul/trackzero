# Ablation Study: Standard Benchmark Results

## Data Size Scaling

All models: 512x4, mixed action type, 40 epochs training.

| N trajectories | Benchmark MSE | vs 10k | step MSE | random_walk MSE |
|---------------|--------------|--------|----------|-----------------|
| 1,000 | 6.30e-2 | 8.2x | 2.22e-1 | 1.12e-1 |
| 2,000 | 1.66e-2 | 2.1x | 5.05e-2 | 3.62e-2 |
| 5,000 | 1.12e-2 | 1.4x | 4.70e-2 | 1.56e-2 |
| 10,000 | 7.72e-3 | 1.0x | 3.57e-2 | 8.56e-3 |

Log-linear scaling: 10x more data -> ~8x better performance.
Diminishing returns above 5k trajectories.

Note: These ablation models used only 40 epochs and batch_size=8192.
The Stage 1C random baseline (100 epochs, batch=65536) scores 2.67e-3 with same data.

## Action Type Ranking

All models: 10k trajectories, 512x4, 40 epochs.

| Action Type | Benchmark MSE | vs mixed | Description |
|-------------|--------------|----------|-------------|
| mixed | 2.50e-2 | 1.0x | Combination of all types |
| multisine | 8.48e-2 | 3.4x | Sum of sinusoids |
| ou | 1.01e-1 | 4.0x | Ornstein-Uhlenbeck process |
| uniform | 2.27e-1 | 9.1x | Uniform random torques |
| gaussian | 2.61e-1 | 10.4x | Gaussian random torques |
| bangbang | 6.35e-1 | 25.4x | Only +/-tau_max |

Action diversity is critical: bangbang is 25x worse than mixed.
Bangbang creates narrow state-space coverage (only extreme actions).

## Key Insights

1. **Data size**: Log-linear returns. More data always helps but diminishes.
2. **Action diversity**: Mixed >> any single type. 25x range between best/worst.
3. **Training setup matters**: Same data with better hyperparams (more epochs,
   larger batch) gives 2.9x improvement (7.72e-3 -> 2.67e-3).
4. **Step/random_walk dominate**: The "hard" families account for 93%+ of error
   regardless of data size or action type.
