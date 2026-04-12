# Why Step and Random Walk Are Hard

## The Misconception
Hard benchmark families (step, random_walk) are NOT out-of-distribution.
Only 2.8–5.8% of their states fall outside the training range.

## The Real Cause: Tail-Heavy Visitation

| Family | q_std | v_std | % Outside Training |
|--------|-------|-------|-------------------|
| multisine | 2.19 | 2.75 | 0.0% |
| chirp | 2.25 | 3.56 | 0.0% |
| sawtooth | 1.97 | 2.65 | 0.0% |
| pulse | 1.58 | 1.95 | 0.0% |
| **step** | **4.71** | **3.42** | **2.8%** |
| **random_walk** | **5.10** | **3.05** | **5.8%** |
| Training data | 2.31 | 2.72 | — |

Step/random_walk have **2× higher position variance** than easy families.
They spend disproportionate time in the tails of the training distribution,
where training samples are sparse.

## Why This Explains the Ranking

**Active learning (#1)**: Queries uncertain regions (tails) → better tail coverage
**Hybrid_select (#2)**: Combines diverse data sources → broader effective coverage
**Rebalance (#3)**: Explicitly reweights data toward underrepresented regions
**Random (#4)**: Dense in mode, sparse in tails
**Maxent_rl (#9)**: Spreads data uniformly but too thinly → poor everywhere

## Implication
The benchmark is really testing **tail accuracy** — how well the model
handles rare-but-physically-valid states. This is the core challenge
for inverse dynamics in real deployment, where the robot encounters
unexpected configurations.

## Per-Family Correlation with Aggregate
- step: r = 0.989
- random_walk: r = 0.992
- All easy families: r < 0.3

The hard families **completely determine** the benchmark ranking.
Easy families are essentially solved by all methods (MSE < 3e-4).
