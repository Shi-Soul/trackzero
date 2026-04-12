# Synthesis: Cross-Stage Findings

## Central Thesis Verdict

**TRACK-ZERO works.** A policy trained entirely on random physics
rollouts — no human demonstrations, no task reference, no domain
knowledge — tracks arbitrary reference trajectories with 14× oracle
MSE. This is 13× better than the best supervised baseline (190×
oracle), confirming that physics-only data generation is viable.

## Key Findings

### 1. Random exploration is a strong baseline

All 9 exploration methods tested in Stage 1C achieve aggregate MSE
within 1.5× of random rollouts at the same data budget (10K
trajectories). The best method (active variance-based filtering)
improves only 1.44× over random. This suggests that the information
content of random-torque data is high, and selective filtering offers
diminishing returns.

### 2. Performance is dominated by hard families

Across all methods, aggregate MSE is dominated by step and
random_walk families. On smooth families (multisine, chirp, sawtooth,
pulse), most TRACK-ZERO methods achieve 0.5–3× oracle. On step and
random_walk, the best method achieves 23–62× oracle. The remaining
gap is attributable to high-velocity tail states that are sparsely
covered even in random rollouts.

### 3. Data quantity >> exploration strategy

Preliminary scaling results show that doubling data from 10K to 20K
trajectories reduces val-loss by 6×, while the best exploration
strategy at 10K improves only 1.5×. This is the central insight:
at the current regime, brute-force data scaling is far more effective
than intelligent data selection.

### 4. Maximum coverage is not optimal

MaxEnt RL achieves the highest state-space coverage but ranks
9th on the benchmark (129× oracle). High coverage pushes the
policy to learn irrelevant extreme states at the expense of
accuracy in the benchmark-relevant regions. The optimal strategy
achieves high coverage with *sufficient density in relevant regions*
— which random rollouts approximate better than entropy-maximizing
policies.

### 5. DAgger fails in this setting

DAgger's iterative on-policy data collection catastrophically
amplifies compounding errors, producing models with 3300× oracle
MSE. This is not a hyperparameter issue — it is a fundamental
mismatch between DAgger's assumption (the expert can correct small
errors) and the inverse dynamics problem (small state errors → large
torque errors in high-Coriolis regions).

## Hypothesis Status (from README)

| # | Hypothesis | Status | Evidence |
|---|-----------|--------|----------|
| H1 | Random rollout covers enough of the state space | ✅ Confirmed | 14.4× oracle aggregate vs 190× supervised |
| H2 | Coverage-driven exploration beats random | ❌ Rejected | Best method only 1.44× better than random |
| H3 | More data beats smarter exploration | 🔄 Testing | Val-loss scaling strongly supportive; benchmark pending |
| H4 | Architecture scaling matters at fixed data | 🔄 Testing | HP sweep experiments in progress |
| H5 | DAgger bridges the gap via on-policy data | ❌ Rejected | 3300× oracle; compounding error makes DAgger unusable |
| H6 | Final model approaches oracle (<2× on all families) | 🔄 Testing | Requires 50K+ data scaling results |

## Remaining Work

The critical experiment is data scaling to 100K trajectories with
the 1024×6 architecture. If the power-law scaling observed in
val-loss extends to benchmark MSE, we predict:

- 20K: ~5× oracle aggregate
- 50K: ~2–3× oracle aggregate
- 100K: potentially near-oracle on all families

These predictions are speculative and await benchmark confirmation.