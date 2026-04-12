# Stage 1A: Supervised Baseline

Train inverse dynamics from multisine dataset. MLP 512×4, 10K trajectories.

## Benchmark Result

| Family | MSE | ×Oracle |
|--------|-----|---------|
| multisine | 1.90e-4 | 1.9× |
| chirp | 2.08e-4 | 0.7× |
| sawtooth | 2.99e-4 | 1.5× |
| pulse | 5.98e-5 | 0.5× |
| step | 1.18e-1 | **364×** |
| random_walk | 9.23e-2 | **2056×** |
| **Aggregate** | **3.52e-2** | **190×** |

Easy families: near-oracle. Hard families: catastrophic.
The model only knows multisine dynamics — step and random_walk
visit states the training data never covered.

## Role in the Project

This is the ceiling on what dataset-dependent training can do.
TRACK-ZERO's value = beating this on hard families without needing
a curated reference dataset.

## Stage 1A Status: ✅ COMPLETE