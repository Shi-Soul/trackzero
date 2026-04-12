# Stage 0: Infrastructure & Baseline

## Deliverables (all complete)

1. **Simulator**: Double pendulum, RK4 integration, realistic inertia/damping
2. **Oracle**: Finite-difference inverse dynamics, benchmark MSE = 1.85e-4 aggregate
3. **Standard benchmark**: 6 reference families × 100 trajectories = 600 total
4. **Coverage metric**: 4D joint histogram over (q1, q2, v1, v2), 10 bins/dim = 10K bins

## Oracle Benchmark (Performance Ceiling)

| Family | Oracle MSE |
|--------|-----------|
| multisine | 1.01e-4 |
| chirp | 3.16e-4 |
| sawtooth | 1.93e-4 |
| pulse | 1.29e-4 |
| step | 3.25e-4 |
| random_walk | 4.49e-5 |
| **Aggregate** | **1.85e-4** |

All subsequent results are measured as multiples of this ceiling.

## Benchmark Families

- **Easy** (multisine, chirp, sawtooth, pulse): smooth oscillatory references
- **Hard** (step, random_walk): sustained extreme states, tail-heavy visitation

Hard families spend 2× more time in high-velocity tails where Coriolis
forces are large. They test tail accuracy, not exotic states — only
2.8-5.8% of test states fall outside the random training distribution.

## Stage 0 Status: ✅ COMPLETE