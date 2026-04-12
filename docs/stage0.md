# Stage 0: Infrastructure & Baseline

> README Goal: Simulator, reference generator, inverse dynamics oracle,
> evaluation harness, logging/visualization.

## Deliverables

| Component | Implementation | Status |
|-----------|---------------|--------|
| Simulator | Double pendulum, RK4 integration, realistic inertia/damping | ✅ |
| Reference generator | Multisine (sum-of-sinusoids), 10K trajectories | ✅ |
| Inverse dynamics oracle | Finite-difference from known EoM | ✅ |
| Evaluation harness | Closed-loop tracking, 6 reference families | ✅ |
| Visualization | State-space coverage plots, trajectory playback | ✅ |

## Oracle Baseline

The inverse dynamics oracle computes exact torques from known equations
of motion. On the multisine test set:

| Policy | Tracking MSE | Notes |
|--------|-------------|-------|
| Oracle (FD) | 3.43e-9 | Near machine-epsilon |
| Supervised MLP | 1.96e-2 | 5.7M× worse |
| Zero torque | 3.27e-1 | Gravity-only baseline |

The oracle is the performance ceiling for all subsequent stages.

## Standard Benchmark

The evaluation harness uses 6 reference families × 100 trajectories = 600 total:

- **Easy**: multisine, chirp, sawtooth, pulse
- **Hard**: step, random_walk

Oracle aggregate MSE on this benchmark: **1.85e-4**.

## Completion Criteria (from README)

- ✅ Oracle achieves near-zero tracking error on all dataset trajectories
- ✅ Evaluation harness produces consistent, reproducible metrics
- ✅ Reference dataset covers broad state space (verified by coverage analysis)

**Stage 0 is COMPLETE.**
