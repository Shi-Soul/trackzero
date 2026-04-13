# Stage 3: Scaling to Higher-DOF Chains (Preliminary)

## Research Question

Does the TRACK-ZERO recipe (random rollout data + cosine LR + WD)
transfer to higher-dimensional articulated systems? How do data
requirements and the oracle gap scale with DOF?

## Experimental Setup

### N-Link Planar Chain

Parametric chain system: N revolute joints in series, each with
identical link parameters (mass, length, inertia, damping, torque limits)
matching the double pendulum configuration.

| System | nq (DOF) | State dim | Action dim | Hidden dim |
|--------|----------|-----------|------------|------------|
| 2-link | 2 | 4 | 2 | 1024 |
| 3-link | 3 | 6 | 3 | 768 |
| 5-link | 5 | 10 | 5 | 1024 |

All use 6-layer MLPs, cosine LR, WD=1e-4 (same recipe as Stage 1 best).
Training data: 10K random rollout trajectories (500 steps each).
Hidden dim = min(1024, 256 × nq).

### Benchmark

Three reference families: uniform random, step (piecewise constant torque),
chirp (frequency sweep). 50 evaluation episodes per family.

### Oracle

Analytical inverse dynamics via MuJoCo `mj_inverse` with finite-difference
acceleration estimate. Same oracle method for all systems.

## Results

### Full Scaling Table (All 1024×6, 10K trajectories)

| System | MLP AGG | Oracle AGG | MLP/Oracle | Params |
|--------|---------|-----------|------------|--------|
| 2-link | 6.80e-4 | 7.63e-5 | 8.9× | 5.26M |
| 3-link | 4.37e-2 | 1.72e-3 | 25.4× | 5.26M |
| 5-link | 1.41e-1 | 5.87e-2 | 2.4× | 5.27M |

### Per-Family Breakdown

**3-link (1024×6):**

| Family | MLP | Oracle | MLP/Oracle |
|--------|-----|--------|------------|
| uniform | 1.57e-4 | 1.33e-4 | 1.2× |
| step | 1.30e-1 | 4.57e-3 | 28.4× |
| chirp | 8.99e-4 | 4.66e-4 | 1.9× |

**5-link (1024×6):**

| Family | MLP | Oracle | MLP/Oracle |
|--------|-----|--------|------------|
| uniform | 3.46e-2 | 2.15e-3 | 16.1× |
| step | 2.97e-1 | 1.34e-1 | 2.2× |
| chirp | 9.24e-2 | 4.02e-2 | 2.3× |

### Finding 1: Model Capacity Is Critical

A controlled ablation on 3-link (768 vs 1024 hidden, all else equal):

| Hidden dim | AGG MSE | Params | vs Oracle |
|------------|---------|--------|-----------|
| 768 | 2.96e-1 | 2.97M | 172× |
| 1024 | 4.37e-2 | 5.26M | 25.4× |

The 1024 model is **6.8× better** overall. Per-family:
- uniform: 443× better (6.96e-2 → 1.57e-4)
- chirp: 109× better (9.80e-2 → 8.99e-4)
- step: 5.5× better (7.20e-1 → 1.30e-1)

Conclusion: model capacity must scale with DOF. The `min(1024, 256*nq)`
heuristic underallocated capacity for 3-link.

### Finding 2: Oracle Degrades Faster Than MLP With DOF

| Metric | 2→3 link | 2→5 link |
|--------|----------|----------|
| Oracle degradation | 22.6× | 770× |
| MLP degradation | 64× | 208× |

The finite-difference oracle (mj_inverse with qacc=(v'-v)/dt) becomes
increasingly inaccurate as DOF and dynamic coupling increase. The MLP's
smooth function approximation degrades more gracefully.

**Caveat**: Oracle degradation partly reflects the finite-difference
approximation quality, not a fundamental limitation of analytical methods.
Newton shooting would give a better oracle baseline.

### Finding 3: Step Family Dominates the Gap

Across all DOF levels, the step family contributes most to the oracle gap:
- 2-link: step is 88% of aggregate error
- 3-link: step is 99.6% of aggregate error
- 5-link: step is 70% of aggregate error

Step references involve discontinuous torque changes, creating the hardest
tracking scenarios for smooth function approximators.

## Interpretation

The TRACK-ZERO recipe (random rollout + cosine LR + WD) transfers to
higher-DOF chains without modification, but performance degrades:
- The MLP/Oracle gap widens from 2→3 links (8.9× → 25.4×)
- But narrows from 3→5 links (25.4× → 2.4×), partly because
  the oracle itself degrades
- Model capacity must scale with DOF (critical finding)

## Next Steps

1. **Algorithm innovation**: multi-step reference conditioning —
   give the policy a window of future reference states
2. **Better oracle**: Newton shooting for higher-DOF systems
3. **Data scaling curve**: how much data does 3-link need to match
   2-link's 8.9× oracle ratio?
