# Stage 1A: Supervised Baseline

> README Goal: Train inverse dynamics from multisine dataset.
> Establishes the baseline that TRACK-ZERO must beat.

## Setup

- Architecture: MLP 512×4 (3.1M params)
- Data: 10K multisine trajectories (8K train / 2K test)
- Input: (current_state, target_next_state) → torques
- Training: 30 epochs, Adam, lr=1e-3

## Results

### In-Distribution (Multisine)

- Val-loss: 1.075e-3 (converged by epoch 29)
- Near-perfect tracking on multisine references

### Standard Benchmark (6 Families)

| Metric | Value |
|--------|-------|
| Aggregate MSE | 3.52e-2 |
| vs Oracle | 190× |
| Easy families (multisine, chirp, sawtooth, pulse) | 1.89e-4 |
| Hard families (step, random_walk) | 1.05e-1 |

### Per-Family Breakdown

| Family | MSE | vs Oracle |
|--------|-----|-----------|
| multisine | 3.09e-5 | 1.6× |
| chirp | 2.71e-5 | 1.4× |
| sawtooth | 6.66e-4 | 3.6× |
| pulse | 3.48e-5 | 1.9× |
| step | 1.53e-2 | 82.8× |
| random_walk | 1.95e-1 | 1054× |

## Key Insight

Supervised learning achieves excellent ID performance but **catastrophic
OOD failure**. The 190× oracle gap is entirely driven by step (83×) and
random_walk (1054×), which the multisine training data never covers.

This is exactly the gap TRACK-ZERO aims to close: learn inverse dynamics
from physics alone, without being limited to a particular motion family.

## Artifacts

- `outputs/stage1a/best_model.pt` — Best checkpoint
- `outputs/supervised_eval.json` — Closed-loop evaluation
