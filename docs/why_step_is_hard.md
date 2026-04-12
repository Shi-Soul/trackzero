# Why Step and Random Walk Dominate the Error

## The 93% Rule

Step + random_walk account for 93-97% of aggregate benchmark error:
- Active: step=68.6%, random_walk=24.8% (total 93.4%)
- Random: step=59.3%, random_walk=37.2% (total 96.5%)
- MaxEnt: step=39.6%, random_walk=37.5% (total 77.1%)

## Root Cause: q2 Distribution Mismatch

| Statistic | Training | Step | Random Walk | Multisine |
|-----------|----------|------|-------------|-----------|
| q2_std | 2.80 | 6.24 (2.2×) | 6.88 (2.5×) | 2.56 |
| % time extreme q2 | 5% | 27.7% | 28.1% | 4.5% |
| % time |v| > 10 | 0.7% | 3.3% | 3.4% | 0.8% |

Step and random_walk generate **large angular excursions** (q2_std 2.2-2.5× wider)
and spend **28% of time in extreme states** (vs 5% for training data).

## Why This Matters for Data Collection

The model must accurately predict torques in these extreme states.
But training data rarely visits them → function approximation is poor there.

**Active learning** wins because it specifically queries uncertain states
(which are exactly these extreme-q2, high-velocity regions).

**DAgger** should win even more because it directly adds training data
from benchmark-like trajectories, covering exactly these regions.

## Implications

To improve benchmark performance, focus exclusively on step/random_walk:
1. Generate training data that visits extreme q2 and high-velocity states
2. Or use DAgger with step/random_walk reference trajectories
3. The "easy" families (multisine, chirp, sawtooth, pulse) are essentially solved
