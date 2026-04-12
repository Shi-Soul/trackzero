# Experiment Status

## Active Experiments

| GPU | Experiment | Config | Status |
|-----|-----------|--------|--------|
| 0 | random_20k_1024x6 | 20K traj, 1024×6, lr=3e-4 | Data gen → training |
| 3 | random_50k_1024x6_v2 | 50K traj, 1024×6, lr=3e-4 | Data gen → training |
| 5 | random_100k_1024x6 | 100K traj, 1024×6, lr=3e-4 | Data gen → training |
| 7 | standard_benchmark_v3 | All models with checkpoints | Evaluating |

## Completed Results

See `outputs/standard_benchmark_v2.json` for the 13-model benchmark.
Top result: active at 10.0× oracle (1.86e-3 aggregate MSE).

## Next Steps

1. When 20K/50K/100K training completes → benchmark all three
2. Plot data-scaling curve: benchmark MSE vs N_trajectories
3. If scaling closes the gap → write Stage 1E conclusion
4. If not → investigate architecture ablations at 100K data