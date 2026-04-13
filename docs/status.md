# Experiment Status

## Active Training (as of benchmark v4 completion)

| GPU | Experiment | Arch | Data | Epoch | Best Val | Est. Remaining |
|-----|-----------|------|------|-------|----------|----------------|
| 0 | random_20k v5 | 1024×6 | 20K | 40/200 | 4.94e-4 | ~16 hrs |
| 1 | random_50k v6 | 1024×6 | 50K | 10/200 | 1.23e-3 | ~30 hrs |
| 7 | random_100k v4 | 1024×6 | 100K | 1/200 | 8.38e-3 | ~80 hrs |

DAgger runs (GPU 5, 7) may still be iterating but produce
consistently bad results (0.6+ aggregate); not worth monitoring.

## Completed Benchmark v4

23 models evaluated. See `outputs/standard_benchmark_v4.json`.
Full ranking in [stage1d.md](stage1d.md).

## GPU Utilization

GPUs 2, 3, 4 have zombie processes (~1–6 GB) but low utilization.
Reclaimable for new experiments.

## Immediate Next Steps

1. Wait for 20K training → benchmark → first scaling data point
2. Launch 512×4 with 20K data (critical: test architecture × data)
3. Kill zombie processes, reclaim GPUs 2–4