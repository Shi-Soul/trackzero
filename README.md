# TRACK-ZERO: Research Execution Plan

## Project Thesis

An optimal whole-body tracking policy can be derived entirely from physics and optimization, without human motion priors. This project builds toward that result incrementally, starting with a double pendulum and scaling to humanoid control.

---

## Stage 0: Infrastructure & Baseline Setup

**Goal:** Establish the simulation environment, evaluation harness, and reference datasets that all subsequent stages build on.

**Deliverables:**

- Double pendulum simulator with configurable torque limits, damping, and time-step. Use MuJoCo or a lightweight custom implementation (Lagrangian dynamics with RK4 integration). The pendulum should have realistic inertial parameters and torque bounds that create a nontrivial reachable set — not everything is achievable from every state.
- Reference dataset generator: drive the pendulum with sampled multisine waves (sum-of-sinusoids with random frequencies, amplitudes, and phases, clipped to torque limits). Simulate forward to collect (state, action, next_state) tuples. Generate a large dataset (e.g., 100k trajectories of 5-10 seconds each) with a held-out test split (80/20). Verify that the dataset has good coverage of the state space by visualizing joint angle/velocity distributions.
- Inverse dynamics oracle: for any feasible trajectory, compute the exact torques via the known equations of motion. This is the performance ceiling.
- Evaluation harness: given a policy and a set of reference trajectories, roll out the policy attempting to track each reference, and report tracking error (MSE on joint angles and velocities over the trajectory). Must support both train-set and test-set evaluation. Also compute per-trajectory statistics (worst-case error, error over time) to detect systematic failure modes.
- Logging and visualization: trajectory playback, tracking error curves, state-space coverage plots.

**Eval criteria for stage completion:**

- The inverse dynamics oracle achieves near-zero tracking error on all dataset trajectories (confirming they are truly feasible).
- The evaluation harness produces consistent, reproducible metrics.
- The reference dataset covers a broad region of the state space (visual inspection + quantitative coverage metric like bin occupancy).

---

## Stage 1: Feasible Reference Tracking — Grokking Inverse Dynamics

**Goal:** Train a universal tracking policy from physics alone that matches the performance of a policy trained on the reference dataset, evaluated on both in-distribution and out-of-distribution feasible references.

This is the core research stage. The agent should iterate extensively, exploring multiple approaches and ablations to find what reliably learns the true inverse dynamics without access to the target dataset.

### 1A: Supervised Baseline (Dataset-Trained)

Train a neural network to approximate inverse dynamics from the multisine dataset. Input: (current_state, target_next_state) → Output: torques. This is straightforward supervised learning and establishes the baseline.

**Eval:** Tracking error on train split and test split of the multisine dataset.

**Expected outcome:** Near-perfect on train, good on test (multisine references are smooth and well-behaved, so generalization within this family should be easy). This baseline is intentionally strong within its distribution.

### 1B: TRACK-ZERO v0 — Random Rollout Self-Supervision

The simplest physics-only approach. No access to the multisine dataset.

- Roll out the pendulum under random torque sequences (uniform random, Gaussian, Ornstein-Uhlenbeck processes of varying correlation times).
- Collect (state, action, next_state) tuples from these rollouts.
- Train the same inverse dynamics network architecture on this self-generated data.

**Eval:** Same evaluation harness, same train/test multisine references. Compare against 1A.

**Iteration targets — the agent should explore:**

- Different random action distributions (white noise, correlated noise, brownian, bang-bang, random multisines with different frequency ranges).
- Dataset size: how much random rollout data is needed to match the supervised baseline?
- State-space coverage analysis: where does the random data concentrate vs. where does the multisine data concentrate? Where does the policy fail?
- Architecture variations: MLP depth/width, whether to condition on a horizon of future reference states vs. just the next state.

**Key question to answer:** Does naive random rollout data achieve full coverage of the feasible state-action space, or does it concentrate in low-energy attractors (hanging down)?

### 1C: TRACK-ZERO v1 — Entropy-Driven Coverage

If 1B reveals coverage gaps (likely), address them with diversity pressure.

**Approaches to explore (iterate across these, don't pick one a priori):**

- **State-space binning with rebalancing:** Discretize the state space into bins. Track bin occupancy during data collection. Preferentially sample initial conditions and action sequences that reach under-visited bins. Retrain on the rebalanced dataset.
- **Maximum entropy exploration:** Formulate data collection as an RL problem where the reward is the entropy of the visited state distribution (or negative log-density under a kernel density estimate). The "policy" here is the data-collection strategy, not the tracking policy.
- **Curiosity / ensemble disagreement:** Train an ensemble of inverse dynamics models. Identify state-action regions where the ensemble disagrees. Prioritize data collection in high-disagreement regions. This directly targets the coverage problem in inverse-dynamics space rather than state space.
- **Hindsight relabeling:** Roll out the current tracking policy on arbitrary references. It will fail on some. Relabel the actual trajectory as a successfully tracked reference. This generates training data in regions the policy naturally visits, and as the policy improves, those regions expand.
- **Adversarial reference generation:** Train a reference generator (a second policy that outputs torque sequences) to maximize the tracker's error. The generator is constrained to the same dynamics, so it can only produce feasible references. This directly finds the hardest feasible references and concentrates training there.

**Eval for each approach:**

- Tracking error on multisine train/test sets (compare to 1A baseline).
- Tracking error on a *new* held-out set of references generated by a different process (e.g., random torques, chirp signals, step functions smoothed to be feasible). This tests true generalization beyond the multisine family.
- State-space coverage metrics before and after the diversity intervention.
- Convergence speed: how many environment steps to reach a given error threshold?

**Iteration protocol for the agent:**

1. Implement approach.
2. Train for a fixed compute budget.
3. Evaluate on all metrics.
4. Analyze failure modes — where does the policy still fail? Is it a coverage problem, a capacity problem, or an optimization problem?
5. Based on the failure mode, either refine the current approach or move to the next one.
6. After exploring all approaches, combine the most effective elements.

### 1D: TRACK-ZERO v2 — Model-Based Exploration with Known Dynamics

Use the simulator as a perfect forward model to drive intelligent exploration, borrowing ideas from model-based RL without the instability of a learned model.

**Approaches to explore:**

- **Reachability-guided sampling:** From a given state, use the known dynamics to compute (or approximate) the set of states reachable in one or a few time steps under admissible torques. Sample reference targets uniformly from this reachable set rather than from the natural dynamics' measure. This directly addresses the diversity problem.
- **Trajectory optimization as data generator:** Use shooting or collocation methods through the known dynamics to generate trajectories that reach specified goal states. Vary the goal states to cover the state space. The resulting trajectories and their optimal torques become training data. This is not the same as the supervised baseline because the goal states are chosen for coverage, not drawn from a particular motion family.
- **Planning-based distillation:** Run online trajectory optimization (iLQR, CEM, MPPI) for each tracking problem at evaluation time. This is expensive but optimal. Then distill the optimizer's behavior into a neural network policy. The question is whether this amortized policy generalizes. No motion data is involved — just physics and optimization.

**Eval:** Same as 1C, plus wall-clock comparison against online trajectory optimization (the policy should be faster at inference while approaching the same quality).

### 1E: Synthesis and Ablation Study

After iterating through 1B-1D, consolidate findings.

- Identify which mechanisms actually mattered (coverage strategy, data distribution, architecture, training procedure).
- Run controlled ablations: each mechanism on/off, measure contribution to final performance.
- Produce a single best TRACK-ZERO configuration for the double pendulum.
- Compare final TRACK-ZERO policy against: (a) supervised baseline on in-distribution references, (b) supervised baseline on out-of-distribution feasible references, (c) analytical inverse dynamics oracle.

**Stage 1 completion criteria:**

- TRACK-ZERO policy matches or exceeds the supervised baseline on in-distribution multisine references.
- TRACK-ZERO policy significantly outperforms the supervised baseline on out-of-distribution feasible references.
- TRACK-ZERO policy approaches the analytical inverse dynamics oracle across a broad range of feasible references.
- Clear understanding of which exploration/coverage mechanisms are essential.

---

## Stage 2: Imperfect and Infeasible Reference Tracking

**Goal:** Extend the TRACK-ZERO policy to handle references that are near-feasible but not exactly achievable, and demonstrate that it learns a principled degradation strategy.

### 2A: Noisy Reference Generation

Take the multisine reference dataset and corrupt it with various noise models:

- Additive Gaussian noise on joint angles/velocities (varying magnitude).
- Temporal perturbations (stretching/compressing time, adding jitter).
- Kinematic perturbations (references generated with different link lengths or mass ratios than the actual pendulum).
- Torque-limit violations (references generated with higher torque limits than the actual system, so they pass through states that require more force than available).

Characterize each corruption by its "infeasibility distance" — how far the corrupted reference is from the nearest feasible trajectory.

### 2B: Optimal Degradation Baseline

For each corrupted reference, solve the tracking problem with online trajectory optimization (the optimizer sees the full reference and the true dynamics/constraints). This gives the optimal compromise for each specific case and serves as the performance ceiling.

### 2C: TRACK-ZERO for Infeasible References

Extend the Stage 1 policy to handle infeasible inputs.

**Approaches to explore:**

- **Train on perturbed feasible trajectories:** Take self-generated feasible data and add the same noise models from 2A. The policy learns that references may not be exactly achievable and must approximate.
- **Feasibility-aware architecture:** Give the policy a mechanism to estimate whether the current reference is feasible and to modulate its behavior accordingly (e.g., predict tracking confidence alongside actions).
- **Multi-step reference conditioning:** Instead of conditioning on just the next target state, condition on a window of future reference states. This lets the policy anticipate upcoming infeasibilities and plan ahead.
- **Adversarial infeasibility training:** The reference generator now produces slightly-infeasible references (e.g., by adding learned perturbations to its own rollouts). The tracker learns to handle them. The generator escalates infeasibility to find the tracker's breaking points.

**Eval:**

- Tracking error vs. infeasibility distance curves — compare TRACK-ZERO against the optimal degradation baseline and the Stage 1 supervised policy.
- Qualitative analysis of degradation strategies — does the policy make reasonable compromises (e.g., tracking position at the expense of velocity, or lagging in time)?
- Robustness: performance variance across different noise types and magnitudes.

**Stage 2 completion criteria:**

- TRACK-ZERO gracefully degrades under increasing reference infeasibility.
- Performance curve is close to the online trajectory optimization baseline.
- Degradation strategy is interpretable and physically reasonable.

---

## Stage 3: Scaling to Articulated Bodies

**Goal:** Validate that the double-pendulum findings transfer to higher-dimensional systems before jumping to full humanoid.

### 3A: Progressively Complex Systems

- 3-link, 5-link, and 7-link planar chains (no contacts).
- 2D biped with ground contact (introduces contact discontinuities).
- 3D floating body with limbs (introduces 3D rotations, higher DoF).

For each system, apply the best TRACK-ZERO recipe from Stage 1/2 and evaluate whether additional mechanisms are needed as dimensionality and contact complexity increase.

### 3B: Contact-Specific Challenges

Contact dynamics are where the real difficulty lies for humanoids. The double pendulum avoids this entirely. Key questions to answer:

- Does the coverage/exploration strategy from Stage 1 naturally discover contact modes (standing, falling, pushing off)?
- Is additional structure needed to handle the hybrid dynamics (continuous phases separated by contact events)?
- How does the adversarial reference generator behave near contact transitions?

### 3C: Scaling Laws

Characterize how data requirements, training compute, and policy capacity scale with system DoF. Establish whether the approach remains tractable at humanoid scale (~30+ DoF).

**Stage 3 completion criteria:**

- TRACK-ZERO works on systems up to ~10 DoF with contacts.
- Clear understanding of how the method scales and what adaptations are needed.
- No fundamental barriers identified for humanoid scale.

---

## Stage 4: Humanoid TRACK-ZERO

**Goal:** Train a humanoid whole-body tracking policy without human motion data and evaluate against mocap-trained baselines.

### 4A: Simulation Setup

- Full humanoid model in MuJoCo (e.g., Unitree H1, or a standard MuJoCo humanoid).
- Mocap-trained baseline (e.g., PHC, AMP, or equivalent) for comparison.
- Human motion reference dataset (AMASS or similar) as the evaluation set — not used for training TRACK-ZERO.

### 4B: TRACK-ZERO Humanoid Policy

Apply the consolidated approach from Stages 1-3. The key is that human motion references are used only for evaluation, never for training.

**Eval:**

- Tracking error on human motion references (in-distribution for the baseline, out-of-distribution for TRACK-ZERO — this is the fair comparison).
- Tracking error on non-human feasible references (out-of-distribution for the baseline, in-distribution for TRACK-ZERO — this is where TRACK-ZERO should win decisively).
- Robustness to perturbations during tracking.
- Generalization to novel motion types not in any mocap dataset.

### 4C: The Definitive Comparison

- TRACK-ZERO vs. mocap-trained baselines on human motions (competitive or close).
- TRACK-ZERO vs. mocap-trained baselines on arbitrary feasible motions (TRACK-ZERO wins).
- TRACK-ZERO vs. online trajectory optimization (TRACK-ZERO is faster, optimization is more precise).

**Stage 4 completion criteria:**

- TRACK-ZERO is competitive with mocap-trained policies on human motion references.
- TRACK-ZERO significantly outperforms mocap-trained policies on out-of-distribution feasible references.
- The result is publishable and the claim is clean: zero human motion data, competitive or superior performance.

---

## Agent Iteration Protocol (All Stages)

Each stage follows the same self-improvement loop:

```
1. IMPLEMENT: Build the current approach variant.
2. TRAIN: Run training for a fixed compute budget.
3. EVALUATE: Run the full eval suite, log all metrics.
4. DIAGNOSE: Analyze where the policy fails.
   - Coverage gap? → Improve exploration/sampling.
   - Capacity gap? → Scale the network or change architecture.
   - Optimization gap? → Adjust learning rate, batch size, training duration.
   - Fundamental gap? → Rethink the approach.
5. HYPOTHESIZE: Form a specific hypothesis about what change will improve results.
6. TEST: Implement the minimal change that tests the hypothesis.
7. COMPARE: Did the metric improve? Was the hypothesis confirmed?
8. RECORD: Log the result, hypothesis, and conclusion.
9. REPEAT from step 5 until diminishing returns, then move to the next approach variant.
```

**Decision criteria for moving between stages:**

- Move from Stage N to Stage N+1 when the completion criteria are met OR when investigation reveals that remaining gaps require the complexity of the next stage to resolve.
- Always prefer depth over breadth within a stage — exhaust the promising ideas before moving on.
- If a later stage reveals a fundamental issue, return to the earlier stage with the new understanding.

---

## Key Hypotheses to Test (Stage 1 Priority)

These are ordered by how informative they are — test the most discriminating hypotheses first.

1. **Random rollout data alone is insufficient for full coverage.** If random torques naturally cover the state space well enough, the whole diversity machinery is unnecessary. Test this first. If it works, the paper is simpler.

2. **Ensemble disagreement is a better coverage signal than state-space density.** Disagreement targets inverse-dynamics uncertainty directly, while state density is a proxy. Compare head to head.

3. **Adversarial reference generation converges to the boundary of the feasible set.** If true, this is the strongest theoretical result. Check whether the generator/tracker game actually reaches equilibrium or oscillates.

4. **Hindsight relabeling provides useful signal even when the tracker is poor.** If early-stage relabeled data is too concentrated near the resting state to be useful, the approach needs warm-starting.

5. **Conditioning on a window of future reference states significantly outperforms single-step conditioning.** This determines whether the policy needs to "plan ahead" or can be reactive.

6. **The gap between TRACK-ZERO and the supervised baseline is larger on out-of-distribution references than on in-distribution references.** This is the core claim of the project — verify it early and quantify the effect size.
