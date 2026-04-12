# Stage 1A: Supervised Baseline

## Research Question

How well can a neural network learn inverse dynamics when trained
directly on the reference dataset? This establishes the performance
ceiling for dataset-dependent methods.

## Setup

- **Training data**: 10K multisine trajectories (8K train / 2K val)
- **Architecture**: MLP 512×4 (4 hidden layers, 512 units each, 3.1M params)
- **Training**: Adam, lr=1e-3, 30 epochs, batch size 65536
- **Input/Output**: (s_t, s_{t+1}^ref) → u_t (2 torques)

## Benchmark Results

| Family | MSE | ×Oracle |
|--------|-----|---------|
| multisine | 1.90e-4 | 1.9 |
| chirp | 2.08e-4 | 0.7 |
| sawtooth | 2.99e-4 | 1.5 |
| pulse | 5.98e-5 | 0.5 |
| step | 1.18e-1 | 364 |
| random_walk | 9.23e-2 | 2056 |
| **Aggregate** | **3.52e-2** | **190** |

## Analysis

The supervised model achieves near-oracle performance on families
that resemble its training distribution (multisine, chirp, pulse,
sawtooth — all smooth oscillatory signals). On step and random_walk,
it fails by 2-3 orders of magnitude because the multisine training
data has near-zero coverage of the sustained high-velocity states
these families visit.

This failure mode defines the problem TRACK-ZERO addresses: a model
trained on any particular reference family will fail on references
outside that family. The only way to achieve universal tracking is to
train on data that covers the full reachable state space — which
requires physics-only data generation (Stages 1B onward).