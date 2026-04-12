# Current Status

## Running Experiments

| GPU | Experiment | Progress |
|-----|-----------|----------|
| 0 | hp_random_1024×6 | ep170/200 |
| 1 | random_50k (1024×6) | ep1/200 |
| 5 | dagger_1024×6 | iter0 ep1 |

## Needs Restart

- **20K random**: crashed (NFS error). Star experiment — val approaching oracle.

## Pending Benchmarks

When training completes, run `scripts/standard_benchmark.py` on:
hp_random_1024×6, random_20k, random_50k, dagger iterations.

## Stage Progress

| Stage | Status |
|-------|--------|
| 0 | ✅ Complete |
| 1A | ✅ Supervised baseline |
| 1B | ✅ Random rollout |
| 1C | ✅ 9 methods benchmarked |
| 1D | 🔄 Scaling experiments |
| 1E | ❌ Needs 1D results |