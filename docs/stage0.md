# Stage 0: Infrastructure & Baseline

## System

Double pendulum with 2 revolute joints, RK4 integration at dt=0.02s.
Torque limits create a nontrivial reachable set — not all state
transitions are achievable.

## Evaluation Protocol

All results in this project use a single evaluation protocol:

**Standard Benchmark.** 100 reference trajectories × 6 signal families
= 600 closed-loop tracking tasks. Each task: given a reference trajectory
r(t), the policy π receives the current state s_t and next reference
state r_{t+1}, outputs torques u_t = π(s_t, r_{t+1}), the simulator
steps to s_{t+1}, and we accumulate tracking error.

**Metric.** Per-trajectory MSE:

    MSE = mean_t[ ||q_t − r^q_t||² + 0.1 · ||v_t − r^v_t||² ]

where q is joint angle, v is joint velocity, 0.1 is the velocity weight
(`configs/medium.yaml:mse_velocity_weight`). The reported number is the
mean MSE across all 100 trajectories in a family ("mean_mse"), or across
all 600 trajectories ("aggregate"). Script: `scripts/standard_benchmark.py`.

**Signal families** (each 100 trajectories, fixed seed=42):

| Family | Generation | Character |
|--------|-----------|-----------|
| multisine | Sum-of-sinusoids (random freq/amp/phase) | Smooth, oscillatory |
| chirp | Frequency-sweeping sinusoid | Smooth, accelerating |
| sawtooth | Piecewise-linear ramps | Smooth, directional |
| pulse | Rectangular pulses with smooth transitions | Mixed dynamics |
| step | Step functions with smoothing | Sustained extreme states |
| random_walk | Brownian motion with smoothing | Persistent tail visitation |

Step and random_walk are qualitatively harder: they hold extreme joint
velocities for extended periods, requiring accurate torque prediction in
high-Coriolis regions where training data is sparse.

## Oracle

The inverse dynamics oracle computes exact torques from the known
equations of motion via finite differences. It represents the
**performance ceiling** — any learned policy can at best match it.

| Family | Oracle MSE |
|--------|-----------|
| multisine | 1.01e-4 |
| chirp | 3.16e-4 |
| sawtooth | 1.93e-4 |
| pulse | 1.29e-4 |
| step | 3.25e-4 |
| random_walk | 4.49e-5 |
| **Aggregate** | **1.85e-4** |

All subsequent results are expressed as both absolute MSE and "×Oracle"
(ratio to oracle aggregate or per-family MSE).

## Coverage Metric

To characterize training data diversity, we define 4D state-space coverage:

    coverage = (occupied bins) / 10,000

where the state space (q₁, q₂, v₁, v₂) is discretized into 10 bins
per dimension (q ∈ [−π, π], v ∈ [−8, 8]), yielding 10⁴ = 10,000 total
bins. We also report normalized Shannon entropy over the bin distribution.
Script: `scripts/search_action_distribution.py:compute_4d_coverage()`.

## Stage 0 Completion

All three README criteria satisfied:
1. Oracle achieves near-zero tracking error (MSE = 3.43e-9 on multisine)
2. Evaluation harness produces consistent, reproducible metrics (fixed seed)
3. Reference dataset covers broad state space (verified by coverage analysis)