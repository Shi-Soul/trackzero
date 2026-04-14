# Research Execution Checklist

Based on README.md proposal with actual progress from docs/ (Stage 0-3).

---

## Stage 0: Infrastructure & Baseline Setup

**Goal:** Establish simulation, evaluation harness, and reference datasets.

### Code Development
- [x] Double pendulum simulator (MuJoCo, RK4 integration, torque limits)
- [x] Reference dataset generator (multisine, configurable)
- [x] Inverse dynamics oracle (finite-difference acceleration)
- [x] Evaluation harness (MSE metric, per-trajectory stats)
- [x] Logging and visualization (trajectories, error curves, coverage plots)

### Research Problems
- [x] Verify oracle achieves near-zero on dataset trajectories
- [x] Verify evaluation metrics are reproducible
- [x] Characterize reference dataset state-space coverage

### Comparative Experiments
- [x] Compute oracle MSE across all 6 signal families

**Status:** ✅ COMPLETE (per stage0.md: all criteria met)

---

## Stage 1: Feasible Reference Tracking — Grokking Inverse Dynamics

**Goal:** Train universal tracking policy from physics alone.

### 1A: Supervised Baseline

#### Code Development
- [x] Train MLP on multisine reference dataset

#### Research Problems
- [x] Measure tracking error on train/test splits

#### Comparative Experiments
- [x] Supervised vs oracle on multisine families
- [x] Per-family error analysis

**Status:** ✅ COMPLETE (stage1.md 1A section)

---

## Stage 1B: TRACK-ZERO v0 — Random Rollout Self-Supervision

#### Code Development
- [x] Random torque rollout data collection (uniform action sampling)
- [x] Train inverse dynamics network on random data

#### Research Problems
- [x] State-space coverage analysis (random vs multisine)
- [x] Coverage-performance relationship

#### Comparative Experiments
- [x] Random rollout vs supervised baseline
- [x] Aggregate tracking error (13× better on random)
- [x] Per-family breakdown (multisine: 20× worse, step/random_walk: 25-34× better)

**Status:** ✅ COMPLETE (stage1.md 1B section)

---

## Stage 1C: TRACK-ZERO v1 — Entropy-Driven Coverage

#### Code Development
- [x] State-space binning with rebalancing
- [x] Maximum entropy exploration (SAC-based)
- [x] Ensemble disagreement tracking
- [x] Hindsight relabeling pipeline
- [x] Adversarial reference generator

#### Research Problems
- [x] Compare effectiveness of 9 coverage strategies
- [x] Identify which mechanisms matter for diversity

#### Comparative Experiments
- [x] Rank all methods vs random baseline (active: 1.44×, DAgger: 0.004×)
- [x] Out-of-distribution generalization test
- [x] State-space coverage metrics before/after
- [x] Convergence speed analysis

**Key findings:**
- Active selection achieves 1.44× improvement (marginal)
- DAgger catastrophically fails (structural incompatibility)
- All methods cluster within 1.5× of random

**Status:** ✅ COMPLETE (stage1.md 1C section)

---

## Stage 1D: TRACK-ZERO v2 — Model-Based Exploration & Training Optimization

### 1D-I: Model-Based Exploration

#### Code Development
- [x] Reachability-guided sampling
- [x] Trajectory optimization as data generator
- [x] Planning-based distillation (CEM MPC)

#### Research Problems
- [x] Compare online planning vs learned policy

#### Comparative Experiments
- [x] Learned policy vs CEM MPC (3,200× better accuracy, 38,000× faster)

**Status:** ✅ COMPLETE (stage2.md 2F section)

### 1D-II: Data Engineering & Scaling

#### Code Development
- [x] Bangbang torque sequences
- [x] Coverage-weighted loss
- [x] Wider initial velocity sampling
- [x] Mixed data scaling (5K→10K→20K→100K)

#### Research Problems
- [x] Test whether data engineering improves results
- [x] Characterize data scaling curves

#### Comparative Experiments
- [x] Bangbang (fails 80×), wider velocity (no effect), weighted loss (fails 1.5×)
- [x] Data scaling: 10K→20K gives 59% improvement
- [x] Phase transition at 50K data

**Status:** ✅ COMPLETE (stage1.md 1D section)

---

## Stage 1D-III: Architecture Ablation

#### Code Development
- [x] Baseline MLP (1024×6, 512×4, 256×3 variants)
- [x] Residual PD architecture (Kp·Δq + Kd·Δv + τ_ff)
- [x] Error-input MLP

#### Research Problems
- [x] Test whether architecture matters more than data

#### Comparative Experiments
- [x] 1024×6 vs 512×4 vs 256×3 (1.29× difference)
- [x] Raw MLP vs Residual PD (9.7× improvement)
- [x] Error-input MLP (lowest val loss, worst benchmark = open-loop gap proof)
- [x] Architecture × Data interaction (Res. PD 2K beats MLP 10K by 14×)

**Status:** ✅ COMPLETE (stage2.md 2A, 2D sections)

---

## Stage 1D-IV: Training Recipe Optimization

#### Code Development
- [x] Cosine learning rate scheduler
- [x] Weight decay tuning
- [x] Huber loss variant
- [x] Dropout experimentation

#### Research Problems
- [x] Find dominant training hyperparameter lever

#### Comparative Experiments
- [x] Cosine LR: 28% improvement (random_walk −47%)
- [x] Weight Decay 1e-4: 9% improvement (step −34%)
- [x] Cosine + WD: 47% improvement (synergistic)
- [x] Huber loss (no benefit), Dropout (harmful 2.5×)

**Status:** ✅ COMPLETE (stage1.md 1D section)

---

## Stage 1E: Synthesis and Ablation

#### Code Development
- [x] Best configuration runner

#### Research Problems
- [x] Consolidate findings across 1A-1D
- [x] Identify essential mechanisms

#### Comparative Experiments
- [x] Final ablation table (data source, exploration, data engineering, architecture, training recipe, data scaling)

**Best Stage 1 Configuration:**
- 1024×6 MLP, Cosine LR + WD=1e-4, 10K random data
- AGG = 4.19e-4 (11,277× oracle)
- Per-family: step ≈ 1000×oracle, random_walk ≈ 14×oracle

**Status:** ✅ COMPLETE (stage1.md summary section)

---

## Stage 2: Imperfect and Infeasible Reference Tracking

**Goal:** Handle noisy/infeasible references, learn degradation strategy.

### 2A: Multi-Step vs Single-Step Reference Conditioning

#### Code Development
- [x] Baseline 1-step conditioning
- [x] 2/4/8-step future reference windows

#### Research Problems
- [x] Test whether multi-step planning helps

#### Comparative Experiments
- [x] K=1 (baseline): AGG = 1.69e-2
- [x] K=8: val_loss = 0.037 (best), AGG = 5.70e-2 (3.4× worse)
- [x] Monotonic degradation with increasing K despite better validation loss

**Key finding:** Open-loop/closed-loop gap — multi-step context hurts closed-loop performance.

**Status:** ✅ COMPLETE (stage2.md 2B section)

---

## Stage 2B: Noise Robustness

#### Code Development
- [x] Noise corruption pipeline (Gaussian on positions/velocities)
- [x] Noise-augmented training

#### Research Problems
- [x] Compare learned policy vs oracle under noise

#### Comparative Experiments
- [x] Clean: MLP 8.9× worse than oracle
- [x] σ=0.01: MLP 5.1× *better* than oracle (crossover point)
- [x] σ=0.05: MLP 2.4× better
- [x] Noise-augmented training: 78× worse on clean, 4.5× better on σ=0.05

**Key finding:** MLP has implicit noise robustness (low-pass filtering effect).

**Status:** ✅ COMPLETE (stage2.md 2C section)

---

## Stage 2C: Residual PD Analysis

#### Code Development
- [x] Residual PD decomposition layer
- [x] Error-coordinate input variant

#### Research Problems
- [x] Test whether physical structure helps

#### Comparative Experiments
- [x] Residual PD: 9.7× better than raw MLP
- [x] Per-family breakdown (Step: 17.3×, Random_walk: 7.9×)
- [x] Error coordinates (lowest val loss, worst AGG = fundamental problem)
- [x] Residual PD + multi-step (K=1: 3.18e-3, K=4: 5.24e-3, always worse with K>1)

**Status:** ✅ COMPLETE (stage2.md 2A, 2E sections)

---

## Stage 2D: CEM MPC Comparison

#### Code Development
- [x] CEM trajectory optimization baseline
- [x] Learned policy speed benchmark

#### Research Problems
- [x] Compare amortized learning vs online planning

#### Comparative Experiments
- [x] Learned policy vs CEM MPC (3,200× better accuracy, 38,000× faster)
- [x] Per-family comparison (multisine: 75,000×, chirp: 60,000×, step: 5,600×)

**Status:** ✅ COMPLETE (stage2.md 2F section)

---

## Stage 2 Summary

**Best Stage 2 Configuration:** Residual PD, single-step, 2K data
- AGG = 1.79e-3
- Findings: Architecture > data (14×), multi-step hurts, error coords catastrophic, learned >> online planning

**Status:** ✅ COMPLETE (stage2.md summary)

---

## Stage 3: Scaling to Articulated Bodies

**Goal:** Test transfer to higher-DOF systems before humanoid.

### 3A: Progressive Complexity (No Contact)

#### Code Development
- [x] 3-link, 5-link chain simulators
- [x] Standard benchmark for each

#### Research Problems
- [x] Test whether Stage 1-2 recipe transfers to higher DOF
- [x] Characterize DOF scaling

#### Comparative Experiments

**Baseline performance vs DOF:**
- [x] 2-link: Raw MLP 2.97e-2, Res. PD 1.79e-3 (16.6× gain)
- [x] 3-link: Raw MLP 1.04e-1, Res. PD 1.58e-2 (6.6× gain)
- [x] 5-link: Raw MLP 2.17e-1, Res. PD 1.35e-1 (1.6× gain)

**Architecture × DOF:**
- [x] PD advantage decays with DOF (cross-joint coupling dominates)
- [x] Full-matrix PD test (worse than diagonal PD on aggregate)
- [x] Structured dynamics hypothesis pending

**Data scaling at high DOF:**
- [x] 5-link: 5× more data (2K→10K) = 0% improvement
- [x] 2-link: 5× data = 1.2× improvement (saturated)
- [x] Bottleneck is expressiveness, not data quantity

**Status:** ✅ COMPLETE (stage3.md 3A section)

---

### 3B: Contact Dynamics

#### Code Development
- [x] Ground plane collision (contact forces, discrete events)
- [x] Contact-aware input feature (binary is_contact flag)
- [x] Per-link contact status variant

#### Research Problems
- [x] Test whether coverage/exploration discovers contact modes
- [x] Test whether additional structure needed for hybrid dynamics
- [x] Quantify contact impact on tracking

#### Comparative Experiments

**2-Link with ground plane:**
- [x] No contact: AGG = 5.68e-4
- [x] Contact baseline: AGG = 7.15e-2 (126× degradation)
- [x] Contact-aware input: AGG = 7.80e-3 (9.2× improvement over baseline)

**Higher DOF with contact:**
- [x] 2-DOF contact-aware: 9.2× gain
- [x] 3-DOF contact-aware: 1.1× gain
- [x] 5-DOF contact-aware: 1.3× gain (essentially no benefit at scale)

**Key finding:** Contact-aware input is breakthrough at 2-DOF but insufficient at higher DOF; scalar flag too coarse for multi-link contact.

**Status:** ✅ COMPLETE (stage3.md 3B section)

---

### 3C: Scaling Laws

#### Research Problems
- [x] Characterize data requirements vs DOF
- [x] Characterize compute requirements vs DOF

#### Comparative Experiments
- [x] ~400× performance degradation from 2-link to 5-link
- [x] Data scaling provides 0% improvement at 5-DOF

**Status:** ⏳ IN PROGRESS (stage3.md interpretation)

---

## Stage 3 Current Investigation

- [ ] Structured dynamics architecture (factored A(q,q̇) @ [Δq; Δv] + b(q,q̇))
- [ ] Prediction: structured advantage should *increase* with DOF

**Status:** ⏳ PENDING (results pending)

---

## Stage 4: Humanoid TRACK-ZERO

**Goal:** Train humanoid policy without human motion data.

### Code Development Required (Not Started)
- [ ] Full humanoid model in MuJoCo
- [ ] Mocap-trained baseline (PHC/AMP equivalent)
- [ ] Evaluation on AMASS mocap dataset

### Research Problems to Address (Not Started)
- [ ] Does approach survive 30+ DOF scale?
- [ ] Contact handling with feet/hands?
- [ ] Full-body coordination beyond 5-link?

### Comparative Experiments Required (Not Started)
- [ ] TRACK-ZERO vs mocap-trained on human motions
- [ ] TRACK-ZERO vs mocap-trained on non-human feasible references
- [ ] Robustness to perturbations
- [ ] Generalization to novel motions

**Status:** ❌ NOT STARTED

---

## Key Hypotheses Status (From README §9)

1. [x] Random rollout data alone is sufficient for high coverage (TESTED & CONFIRMED)
2. [x] Ensemble disagreement beats state-space density (TESTED & PARTIALLY CONFIRMED)
3. [ ] Adversarial reference generation reaches feasible boundary (TESTED PARTIALLY, needs deeper analysis)
4. [x] Hindsight relabeling provides early signal (TESTED & CONFIRMED useful)
5. [x] Multi-step conditioning outperforms single-step (TESTED & REFUTED — multi-step hurts)
6. [x] Out-of-dist gap > in-dist gap (TESTED & CONFIRMED core claim)

---

## Summary

**Completed:** Stages 0, 1 (full), 2 (full), 3A-3B
**In Progress:** 3C (structured dynamics architecture)
**Not Started:** Stage 4

**Major Discoveries:**
- Training recipe (Cosine+WD) is dominant lever in Stage 1 (47% gain)
- Architecture (Residual PD) dominates data in Stage 2 (14× better for 2K vs 10K)
- DOF scaling is fundamental bottleneck: ~400× degradation 2→5-link
- Contact-aware input breakthrough at 2-DOF, collapses at higher DOF
- Open-loop metrics (validation loss) completely unreliable for closed-loop performance

**Critical Blockers for Humanoid:**
1. DOF scaling (expressiveness gap, not data gap)
2. Contact complexity (scalar flags insufficient at higher DOF)
3. Cross-joint coupling (inertia matrix M(q) requires architectural innovation)
