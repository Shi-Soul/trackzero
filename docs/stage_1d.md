# Stage 1D: Model-Based Exploration

## Motivation

Stage 1C showed that **reweighting** random rollout data doesn't help — the
random baseline is unreasonably strong because it already covers the feasible
state space broadly. The key insight: we need to **generate qualitatively new
data**, not just reshuffle existing random rollouts.

Stage 1D uses the **known forward dynamics** (MuJoCo simulator) as a perfect
model to drive intelligent data generation.

## Three Approaches from the Proposal

### 1D.1: Reachability-Guided Sampling

**Idea:** Instead of collecting trajectories under random torque sequences
(which concentrate near low-energy attractors), systematically sample
(state, action) pairs that cover the full feasible region.

**Method:**
1. Grid the 4D state space (q1, q2, v1, v2) at representative points
2. At each grid state, sample actions uniformly in [-τ_max, τ_max]²
3. Simulate one step: s' = f(s, a)
4. Collect (s, a, s') as training data for inverse dynamics

**Key advantage:** This generates data at arbitrary states, not just states
reachable by following random torque sequences from rest. It fills the "gaps"
that random rollouts miss.

**Challenge:** Single-step data lacks trajectory coherence. The model learns
local inverse dynamics but may not generalize to multi-step tracking where
errors compound. We test this directly.

### 1D.2: Trajectory Optimization as Data Generator

**Idea:** Use shooting/CEM/MPPI through the known dynamics to generate
trajectories that reach specified goal states. Vary goals for coverage.

**Method:**
1. Sample random goal states in the feasible region
2. For each goal, run CEM/MPPI to find a torque sequence that reaches it
3. The resulting trajectories become training data
4. Goals chosen to cover the state space uniformly

### 1D.3: Planning-Based Distillation (iLQR/CEM/MPPI)

**Idea:** For each test tracking problem, run online trajectory optimization
to find the optimal torques. Then distill the optimizer's behavior into a
neural network. No motion data — just physics and optimization.

**Method:**
1. Given a reference trajectory, solve for optimal tracking torques online
2. Collect many (reference, optimal_torques) pairs
3. Train a policy to amortize the optimizer
4. Compare: policy vs online optimizer (speed/quality tradeoff)

## Experimental Plan

### Phase 1: Reachability-Guided Single-Step Data (1D.1)

Start here because it's simplest and most different from 1C.

1. Generate N single-step transitions from uniformly sampled states
2. Train MLP on these transitions (same architecture: 512×4)
3. Evaluate on ID multisine and OOD references
4. Compare to random_matched baseline from 1C
5. Test: N=1M, 5M, 10M single-step pairs

### Phase 2: Trajectory Optimization Generator (1D.2)

1. Implement CEM-based trajectory optimizer using known dynamics
2. Generate diverse goal-reaching trajectories
3. Train on resulting data, evaluate

### Phase 3: Planning Distillation (1D.3)

1. Implement CEM/MPPI tracker for reference following
2. Run on eval references, collect optimal torques
3. Distill into MLP, compare amortized vs online

## Expected Outcomes

- 1D.1 may help ID but hurt OOD if single-step data doesn't capture
  trajectory-level dynamics
- 1D.2 should produce high-quality diverse trajectories
- 1D.3 is the gold standard but expensive; tests whether amortization works

## Artifacts

| Output directory | Description |
|---|---|
| `outputs/stage1d_reachability/` | Reachability-guided single-step |
| `outputs/stage1d_trajopt/` | Trajectory optimization generator |
| `outputs/stage1d_distill/` | Planning-based distillation |
