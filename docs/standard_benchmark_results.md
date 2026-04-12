# Standard Benchmark Results

## Benchmark Design

Fixed, method-agnostic benchmark: 600 tracking trajectories across 6 signal families.
Each family contributes 100 trajectories, all deterministically seeded (seed=12345).

| Family | Description | Difficulty |
|--------|-------------|-----------|
| multisine | In-distribution (from test set) | Easy |
| chirp | Frequency-sweeping sinusoids | Easy |
| sawtooth | Ramping torque signals | Easy |
| pulse | Square pulse torques | Easy |
| step | Sudden torque changes → large excursions | Hard |
| random_walk | Cumulative random torques → drift | Hard |

**Single metric**: mean tracking MSE across ALL 600 trajectories.

## Results (11 models)

| Rank | Method | Aggregate MSE | vs Best | step | random_walk |
|------|--------|--------------|---------|------|------------|
| 1 | active | 1.858e-3 | 1.0× | 7.64e-3 | 2.76e-3 |
| 2 | hybrid_select | 2.155e-3 | 1.2× | 8.67e-3 | 3.66e-3 |
| 3 | rebalance | 2.176e-3 | 1.2× | 9.70e-3 | 2.72e-3 |
| 4 | random | 2.666e-3 | 1.4× | 9.49e-3 | 5.94e-3 |
| 5 | density | 3.172e-3 | 1.7× | 1.28e-2 | 5.59e-3 |
| 6 | hybrid_weighted | 4.492e-3 | 2.4× | 1.29e-2 | 1.37e-2 |
| 7 | adversarial | 5.011e-3 | 2.7× | 1.18e-2 | 1.73e-2 |
| 8 | hindsight | 8.130e-3 | 4.4× | 3.32e-2 | 1.45e-2 |
| 9 | maxent_rl | 2.390e-2 | 12.9× | 5.67e-2 | 5.38e-2 |
| 10 | supervised_1a | 3.522e-2 | 19.0× | 1.18e-1 | 9.23e-2 |
| 11 | hybrid_curriculum | 4.509e-2 | 24.3× | 1.37e-1 | 1.33e-1 |

## Key Findings

### 1. Ranking reversal from ID-only evaluation

Old ranking (ID eval): random (#1) > supervised_1a (#2) > hybrid (#3)
New ranking (standard): **active (#1)** > hybrid_select (#2) > rebalance (#3) > random (#4)

The ID-only evaluation was misleading. On the proper task benchmark, **active learning
beats the random baseline by 30%**.

### 2. Step and random_walk dominate the aggregate

The "easy" families (multisine, chirp, sawtooth, pulse) are ~1e-4 for most methods.
The "hard" families (step, random_walk) are 10-100× worse and drive the ranking.

### 3. supervised_1a catastrophically fails on hard families

Despite excellent multisine performance (1.9e-4), supervised_1a scores 0.118 on step
and 0.092 on random_walk — it was overfitting to multisine distribution.

### 4. hybrid_curriculum is the worst model

The curriculum strategy (maxent pretrain → random finetune) that looked promising on
ID eval (3.1e-4) is actually the worst on the benchmark (4.5e-2). The finetuning on
random data catastrophically erased step/random_walk capability.
