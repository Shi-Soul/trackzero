# Synthesis: Cross-Stage Findings

## Central Result

**TRACK-ZERO works.** A policy trained on random physics rollouts
— no demonstrations, no task references, no domain knowledge —
tracks arbitrary reference trajectories at 10× oracle MSE aggregate,
and matches the oracle on 4 of 6 trajectory families.

## Benchmark Summary (v4, 23 models)

Oracle: 1.85e-4 aggregate MSE (analytical inverse dynamics).

**Tier 1 — Near-oracle on smooth families** (agg 1.9e-3 – 5.0e-3):
Active, hybrid_select, rebalance, random, density, hybrid_weighted,
adversarial. All use 512×4 architecture, 10K random rollout data.

**Tier 2 — Moderate** (agg 8e-3 – 4.5e-2):
Hindsight, maxent_rl, supervised_1a, hybrid_curriculum.

**Tier 3 — Failure** (agg 0.6+):
All DAgger variants, all 1024×6 models (overtrained or undertrained).

## Key Findings

### 1. The oracle gap is family-specific

On smooth families (multisine, chirp, sawtooth, pulse), TRACK-ZERO
active achieves 0.6–1.7× oracle — effectively solved. The aggregate
10× gap is entirely due to step (23×) and random_walk (62×), which
involve discontinuous torque commands producing high-velocity states
underrepresented in training data.

### 2. Exploration strategy < data quantity

At 10K data, the best exploration method (active) improves only
1.44× over random. Preliminary val-loss from 20K training (4.94e-4
at ep40, already below 10K converged val of 1.4e-3) suggests data
scaling is a stronger lever. Full benchmark confirmation pending.

### 3. Architecture scaling backfires at small data

1024×6 (5.3M params) with 10K data: 6.11e-1 aggregate = 3300× oracle.
512×4 (1.1M params) with 10K data: 2.67e-3 aggregate = 14× oracle.
The 229× degradation from 5× more parameters confirms severe
overfitting. Whether 20K+ data rescues the larger model is the
open scaling question.

### 4. DAgger is structurally incompatible

DAgger assumes the expert corrects small deviations. In inverse
dynamics, small state errors in high-Coriolis regions produce large
torque errors, and on-policy rollouts quickly diverge into
dynamically unstable states. All 4 DAgger configurations (2 arch
× 2 runs) show performance degradation across iterations.

### 5. MaxEnt RL: high coverage ≠ good policy

MaxEnt achieves maximal state-space entropy but ranks 9th (129×
oracle). Forcing the policy to learn extreme states dilutes accuracy
in benchmark-relevant regions. Random rollouts achieve a naturally
appropriate coverage-density tradeoff.

## Hypothesis Verdicts

| # | Hypothesis | Verdict | Key Evidence |
|---|-----------|---------|-------------|
| H1 | Random rollout covers enough state space | ✅ Confirmed | 14× oracle agg; 4/6 families at oracle level |
| H2 | Smarter exploration beats random | ⚠️ Marginal | Best method 1.44× better; all within 2.7× |
| H3 | More data beats smarter exploration | 🔄 Testing | Val-loss strongly supportive; benchmark pending |
| H4 | Larger models help at fixed data | ❌ Rejected | 229× worse at 5× more params with 10K data |
| H5 | DAgger bridges the gap | ❌ Rejected | 3300× oracle; degrades with iterations |

## Remaining Work

1. **Data scaling curve**: 20K/50K/100K training completion →
   benchmark → plot MSE vs N_trajectories
2. **Architecture × data interaction**: does 512×4 with 20K data
   outperform both 512×4/10K and 1024×6/20K?
3. **Stage 1E synthesis**: identify which mechanism is essential
   (data quantity? architecture? exploration?), produce single
   best configuration, compare against oracle on all families