# Stage 1C: Per-Variant Details and Failure Modes

## 1. Ensemble Disagreement

**Idea:** Train bootstrap ensemble on seed data, score candidates by prediction
variance, select highest-disagreement trajectories.

**Result:** Worst overall. Geomean ratio to random = 1.47x (47% worse).

**Why it failed:**
- Trajectory-level scoring is too blunt — a few chaotic timesteps pull in
  an entire trajectory of mostly-uninformative data.
- Bootstrap ensemble calibration is poor on only 2000 seed points.
- High-uncertainty regions are often hard-to-fit, not useful-to-learn.
- This is the classic "curiosity trap" — uncertainty != usefulness.

**Artifact:** `outputs/stage1c_active_full/`

---

## 2. Bin Rebalancing

**Idea:** Discretize 4D state space (q1,q2,v1,v2) into bins. Score each
trajectory by how many rare bins it visits. Select trajectories that most
expand coverage of under-visited regions.

**Result:** Mixed. Slightly helps `random_walk` (0.92x) but catastrophically
hurts `mixed_ood` (6.15x worse).

**Why it failed:**
- Forcing uniform coverage over (q,v) space wastes data budget on
  physically extreme states that rarely matter for tracking.
- The 10-bin-per-dimension grid is too coarse — rare bins may span
  very different dynamics regimes.
- Rebalancing sacrifices representation of common (important) states.

**Artifact:** `outputs/stage1c_rebalance_full/`

---

## 3. Low-Density / Max-Entropy Proxy

**Idea:** Score each trajectory by average inverse local occupancy in 4D bins.
Selects trajectories that visit the least-populated regions overall.

**Result:** Best of the selection methods. Helps `sawtooth` (0.81x ratio).
Still hurts `mixed_ood` (3.27x).

**Why partial success on sawtooth:**
- Sawtooth references produce moderate, smooth state excursions.
- Low-density selection pushes data into these moderate-energy regions
  that random rollouts under-represent.
- But `mixed_ood` includes extreme states where the density-pushed data
  has insufficient representation.

**Artifact:** `outputs/stage1c_density_full/`

---

## 4. Hybrid Coverage

**Idea:** Z-score normalized combination of low-density and rebalancing scores.

**Result:** Helps `sawtooth` (0.91x) and `pulse` (0.95x). Hurts `mixed_ood`
(3.46x) and `random_walk` (1.40x).

**Interpretation:** Averaging two mediocre signals doesn't produce a good one.
The hybrid inherits the worst properties of both: it wastes budget on rare
states (from rebalancing) while under-weighting common states (from density).

**Artifact:** `outputs/stage1c_hybrid_full/`

---

## 5. Hindsight Relabeling

**Idea:** Roll out the best Stage 1B teacher policy against hard (mixed_ood)
reference trajectories. The teacher will fail to track them perfectly. Relabel
the actually-achieved trajectory as a new "reference" that was successfully
tracked. Train on this relabeled data + seed random data.

**Result:** Worst ID performance (2.4x worse than random). Val loss 0.050 vs
0.002 for random-trained models.

**Why it failed:**
- Teacher has MSE=0.010 on mixed_ood → deviates substantially from references.
- Achieved trajectories cluster near the teacher's "comfort zone"
  (states where it already performs well).
- The relabeled dataset has very different distribution from multisine
  validation data, causing high val loss.
- Single-round hindsight: no iteration to expand the teacher's comfort zone.

**What would help:** Iterative hindsight where the teacher is retrained each
round on accumulated relabeled data, progressively expanding its range.

**Artifact:** `outputs/stage1c_hindsight_full/`

---

## Cross-Cutting Analysis

### Why does random win on mixed_ood?

`mixed_ood` combines all OOD families. It requires broad coverage across the
entire state space. Every selector method sacrifices breadth for depth in some
specific region. Random rollouts provide the most uniform coverage under
the data budget constraint.

### Tail risk

All methods show extreme max/median MSE ratios (1000–20000x) on hard OOD
families, indicating catastrophic failures on a small fraction of test
trajectories. No selection method reduces this tail risk — the problem is
likely model capacity on extreme states, not data coverage.

### What the proposal got right and wrong

The proposal correctly identified coverage as the key challenge. But it
assumed the coverage gap could be closed by smarter data *selection* from
random rollouts. In practice, random rollouts already cover the feasible space
well enough that selection only *redistributes* the fixed budget rather than
*expanding* into uncovered regions.

Stage 1D's model-based approaches (reachability, trajectory optimization)
are the right next step because they can *generate* data in specific target
regions, not just reweight existing random samples.
