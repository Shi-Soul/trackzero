# Experiment Status

## Active Training (All 8 GPUs)

| GPU | Experiment | Arch | Data | Epoch | Best Val |
|-----|-----------|------|------|-------|----------|
| 0 | 20K random (v5) | 1024x6 | 20K | 70/200 | 3.09e-4 |
| 1 | 50K random (v6) | 1024x6 | 50K | 20/200 | 4.54e-4 |
| 2 | 20K random | 512x4 | 20K | 20/200 | 3.18e-3 |
| 3 | 50K random | 512x4 | 50K | data gen | - |
| 4 | bangbang aug | 512x4 | 10K | 40/200 | 4.50e-3 |
| 5 | bangbang aug | 1024x6 | 10K | 1/200 | 2.01e-2 |
| 6 | 100K random | 512x4 | 100K | data gen | - |
| 7 | 100K random (v4) | 1024x6 | 100K | 10/200 | 8.10e-4 |

Zombie processes cleaned: 47 killed, freeing GPUs 3/5/6.
Snapshot of ep140 checkpoint (val 1.70e-4) in outputs/snapshots/.

## Experiment Matrix

|  | 512x4 (1.1M) | 1024x6 (5.3M) |
|--|--------------|----------------|
| 10K random | done (baseline) | done |
| 20K random | GPU2 | GPU0 |
| 50K random | GPU3 | GPU1 |
| 100K random | GPU6 | GPU7 |
| 10K bangbang | GPU4 | GPU5 |

## Key Pending Results

1. Data scaling curve: does 10K to 100K close the 10x oracle gap?
2. Bangbang vs random: does targeted high-velocity coverage
   outperform brute-force data scaling at fixed 10K?
3. Architecture interaction: does 1024x6 benefit more from
   scaling than 512x4, or does it overfit?
