# Learned HLP spot-check findings (April 2026)

## Question
Before committing to step 2 (perturbation rollouts to label hindsight-optimal
HLP allocations), is there enough learnable signal to justify the work?

## Method
For 5 representative configs across saturation regimes, capture the
heuristic's allocation at each rebalance tick. For each tick, generate
N random per-region perturbations (±k vehicles between two random
regions, sum-preserving) and re-run the day with the perturbation
applied at that single tick. Measure end-of-day assignment-rate delta.

## Results (seed=11, OSM Memphis, n_outskirts=25)

### ±2 magnitude, 5 perturbations per tick (15 ticks total)

| regime | ticks beatable (>+0.5pp) | max gain |
|--------|--------------------------|----------|
| light_load (fleet=60, int=1, K=5) | 0/3 | 0 |
| constrained_lowK (60, 3, K=3) | 2/3 | +2.31 pp |
| constrained_highK (60, 5, K=5) | 2/3 | +1.05 pp |
| moderate (120, 3, K=5) | 0/3 | 0 |
| heavy_saturation (120, 5, K=5) | 0/3 | 0 |

Beatable: 4/15 = 26.7%. Max gain: +2.31pp.

### ±5 magnitude, 10 perturbations per tick (constrained + heavy only)

| regime | ticks beatable | max gain |
|--------|----------------|----------|
| constrained_highK | 2/3 | +3.14 pp (tick 0) |
| heavy_saturation | 1/3 | +1.05 pp (tick 2) |

Larger perturbations escape local minima at one tick (constrained_highK
tick 0 went 0 → +3.14pp), but the broader pattern persists.

## Findings

1. **Signal lives only at constrained corners.** Light and moderate loads
   show zero learnable signal — the heuristic's proportional-to-demand
   allocation is already optimal there.
2. **Gains where present are small** (1-3pp at best, often 0).
3. **Tick 0 dominates.** Later ticks usually have 0 signal once the
   route system has settled into a local optimum from the first allocation.
4. **The heuristic uses a perfect demand oracle**, which gives it most of
   the win. A learned HLP at this scale (5 hubs, 5 regions, 25 outskirts)
   has little extra information to exploit.

## Decision

**Do not pursue learned HLP at v1 scale.** Cost (perturbation rollouts +
training + integration) is high; expected gain is 0-3pp at constrained
corners, ~0pp elsewhere. The path we already have (`main`):

  - hierarchical decomposition (heuristic HLP):      ~4× speedup over mono+heur
  - + learned LLP filter:                            ~12× compounded at scale
  - quality at parity with heuristic baseline

is dominant.

**Where learned HLP would actually earn its keep** (and is the right v2 axis):
  1. **Multi-modal** mix (microtransit + bus feeders + AV mix). Adds modes
     to the rebalance action space, where heuristic averaging loses information.
  2. **Stochastic travel times**. Breaks the heuristic's perfect-forecast
     assumption — model can learn risk-aware allocation.
  3. **More zones / hubs**. Heuristic's "proportional to summed demand"
     averaging degrades as state-space grows.
  4. **Longer / multi-step horizons**. Heuristic only looks 60 min ahead;
     a learned policy could learn shift-to-shift coupling.

## Artifacts

- `scripts/hlp_spot_check.py` — reproduces the spot-check.
- Step 1 infrastructure (`src/hcoord/learning/hlp_features.py`,
  `hlp_collector.py`) is left in place: when v2 conditions are right
  (multi-modal / stochastic), the data pipeline is ready.
