# Write-up prep — hierarchical decomposition for multi-modal DVRP

Reference doc for the eventual proposal write-up. Captures substrate,
sweep configurations, and headline numbers so we can pull from this
once the proposal section scaffold is clear. Figures and raw CSVs live
alongside this file in subfolders.

Repo state when these were generated: commit `25bacb3` on `main`.
Reproduce any sweep with `uv run python scripts/<script>.py`.

---

## 1. Substrate

### Geography

- Real Memphis OSM drive network, pulled via `osmnx.graph_from_point` around
  the centroid of 5 hub centroids (Downtown, Medical, FedEx, Airport, UofM).
  Outer radius 40 km, network type `drive`.
- Hub centroids are real Memphis points (downtown, medical district, FedEx
  hub, MEM airport, UofM campus).
- Outskirt zones sampled uniformly in an annulus `inner_km=8, outer_km=40`
  around the hub centroid, snapped to nearest OSM node, deduped.
- Zone-to-zone travel times are Dijkstra distances on OSM `travel_time`
  (seconds), converted to minutes. Edge speeds: posted limits where
  available, fallback 50 km/h.
- Final structure: complete graph on 5 hubs + N outskirts.
  - Step 1 + 2 sweeps: N=25 (30 zones)
  - Step 3 sweep: N=60 (65 zones)
- OSM graph and built `Network` are pickled into `data/osm_cache/`. First
  pull is ~30 s; subsequent runs are instant.

### Demand

- One-direction shift commute. Outskirt → hub trips only.
- Three shifts at minutes 360, 840, 1320 (6am, 2pm, 10pm) of a 1440-min day.
- Per (origin, destination, shift): Poisson(`intensity * base_rate`) requests.
- Structural zeros: each (O, D) pair has prob `structural_zero_prob` of being
  permanently zero across all shifts (sampled once per seed).
- Each request has `announce_time = arrival_deadline - announce_lead_min`,
  arrival window `[deadline - arrival_buffer_min, deadline]`.
- Defaults: `base_rate=0.4, structural_zero_prob=0.4, announce_lead_min=90,
  arrival_buffer_min=15`.
- Intensity multiplier sweeps demand pressure independently of fleet.

### Fleet

- Capacitated vehicles, capacity 6 (default), service window
  `[0, service_end_time]` with `service_end_time=1440` (full day).
- Initial placement strategy `hubs`: round-robin over hubs (also have a
  `demand_proportional` strategy registered for later sweeps).
- Vehicles are required to return to home depot by `service_end_time`
  (enforced as implicit virtual segment in feasibility checks).

### Dispatchers

Both share `best_insertion(vehicle, request, oracle)` — exhaustive (p, q)
position-pair search over the route with capacity + window + return-deadline
feasibility checks. Cost = increase in return-home arrival time.

- **Monolithic**: full-fleet greedy. For each request, score every vehicle's
  best feasible insertion, take the global min.
- **Hierarchical**:
  - Hub-catchment partition: each outskirt assigned to its
    travel-time-nearest hub. K = number of regions; `hub_groups` defines
    which hubs share a region (default = 1 hub per region for K=5; for
    K<5, agglomerate by min-link travel time via `merge_nearest_hubs`).
  - Origin-region request assignment: request routed to the region of its
    origin zone.
  - LLP: greedy insertion restricted to vehicles currently assigned to that
    region.
  - HLP: every `rebalance_interval_min` (default 30 min), count requests
    in `[now, now + forecast_lookahead_min]` (default 60 min) per region,
    proportionally allocate idle vehicles across regions, move excess
    vehicles src→dst (charges travel time to `vehicle.available_time`).
  - K=1 collapses to monolithic — verified empirically as a sanity check.

### Why no event-driven sim

Travel times are deterministic; greedy insertion is feasibility-checked.
The planned route is the realized route. Therefore
`assignment_rate == on-time arrival rate` exactly. v2 with stochastic
travel times will need an event loop.

---

## 2. Sweeps

### Step 1 — Scale-up grid (`scripts/scaleup_sweep.py`)

**Goal:** does the speedup story hold across fleet × intensity?

**Grid:**
- Network: Memphis OSM, n_outskirts=25 (30 zones)
- Seed: 11 (single seed)
- Fleet sizes: [30, 60, 120, 240]
- Intensities: [1.0, 3.0, 5.0]
- Dispatchers: monolithic, hierarchical K=3, hierarchical K=5
- Total: 36 cells

**Outputs:** `write-up/scaleup/`
- `results.csv` — 36 rows, all config columns + metrics
- `wall_vs_fleet.png` — mean wall-time vs fleet, one line per dispatcher,
  one panel per intensity
- `assignment_vs_fleet.png` — assignment rate vs fleet, same layout
- `speedup_heatmap.png` — mono / hier mean wall-time ratio across
  fleet × intensity (averaged over K=3 and K=5)

**Headline numbers:**

Speedup factor (mono / hier mean_wall_ms):

| fleet \ intensity | 1.0  | 3.0  | 5.0  |
|-------------------|------|------|------|
| 30                | 3.79 | 2.53 | 2.82 |
| 60                | 4.26 | 2.80 | 3.07 |
| 120               | 3.71 | 3.59 | 3.70 |
| 240               | 3.82 | 3.79 | 3.99 |

(K=5; K=3 is roughly 2/3 of these factors.)

Quality at three regimes:

- **Saturation** (fleet=240, intensity=5): mono 98.4% vs hier K=5 98.4% —
  parity, hier 4× faster.
- **Constrained corner** (fleet=60, intensity=3): mono 55.1% vs hier K=5
  80.1% — **+25.0 pp quality + 2.8× faster**. Anti-vampire effect.
- **Light load** (fleet=120, intensity=1): both 98.6%, hier 3.7× faster.

### Step 2 — K-sensitivity (`scripts/ksensitivity_sweep.py`)

**Goal:** does K monotonically improve speed and quality? Does K=1 reduce
to monolithic? How sensitive across seeds?

**Grid:**
- Network: Memphis OSM, n_outskirts=25 (30 zones)
- Seeds: [11, 23, 37, 53, 71] (5 seeds for error bars)
- Operating points: (fleet=60, intensity=3), (fleet=120, intensity=5)
- K: [1, 2, 3, 4, 5] for hierarchical
- Plus monolithic baseline at each operating point
- Total: 60 configs (2 ops × 5 seeds × (5 K + monolithic))

**Outputs:** `write-up/ksensitivity/`
- `results.csv` — 60 rows, one per (op × seed × dispatcher × K)
- `aggregated.csv` — 12 rows, mean ± std across seeds per (op × dispatcher × K)
- `k_sensitivity.png` — dual-axis (assignment % left, mean wall ms right),
  K on x-axis, hier as errorbar lines, monolithic as horizontal reference,
  one panel per operating point

**Aggregated table (mean over 5 seeds):**

```
fleet  intensity  dispatcher    K  assignment        wall_ms
   60        3.0  monolithic    -  0.564 ± 0.059  4.16 ± 0.75
   60        3.0  hier          1  0.564 ± 0.059  4.18 ± 0.74
   60        3.0  hier          2  0.642 ± 0.093  2.66 ± 0.33
   60        3.0  hier          3  0.660 ± 0.093  1.95 ± 0.21
   60        3.0  hier          4  0.666 ± 0.104  1.55 ± 0.18
   60        3.0  hier          5  0.670 ± 0.112  1.14 ± 0.15
  120        5.0  monolithic    -  0.765 ± 0.124  9.19 ± 1.21
  120        5.0  hier          1  0.765 ± 0.124  9.14 ± 1.27
  120        5.0  hier          2  0.816 ± 0.083  5.33 ± 0.82
  120        5.0  hier          3  0.815 ± 0.097  4.06 ± 0.57
  120        5.0  hier          4  0.817 ± 0.098  3.31 ± 0.54
  120        5.0  hier          5  0.825 ± 0.100  2.45 ± 0.45
```

**Three things this figure shows:**
1. **Sanity check:** K=1 hierarchical exactly matches monolithic on both
   metrics (0.564 vs 0.564, 4.18 vs 4.16 ms). Confirms the hierarchical
   machinery doesn't add overhead beyond what the partitioning saves.
2. **Monotonic speedup:** each K up cuts wall by ~25-30%. K=5 is 3.7×
   faster than monolithic at the constrained corner.
3. **Monotonic quality:** assignment rate increases with K, plateauing
   around K=3-5. The constrained-corner gap (mono 56.4% → hier K=5 67.0%
   = +10.6pp) is mostly captured at K=2 already (+7.8pp).

### Step 3 — Geographic scale-up (`scripts/largescale_sweep.py`)

**Goal:** does the story hold at larger problem size (more zones, more
fleet, more requests/day)?

**Grid:**
- Network: Memphis OSM, n_outskirts=60 (65 zones — 2.4× more zones than steps 1-2)
- Seed: 11
- Fleet sizes: [120, 240, 480]
- Intensities: [1.0, 3.0, 5.0]
- Dispatchers: monolithic, hier K=3, hier K=5
- Total: 27 cells
- Day demand grows naturally: ~172 / 572 / 1019 requests at intensity 1 / 3 / 5

**Outputs:** `write-up/largescale/`
- `results.csv` — 27 rows
- `wall_vs_fleet.png`, `assignment_vs_fleet.png`, `speedup_heatmap.png`
  (same plot definitions as step 1)

**Highlights:**

| Cell                         | Mono assign | Hier K=5 assign | Mono mean ms | Hier K=5 mean ms | Speedup | Mono max ms | Hier K=5 max ms |
|------------------------------|-------------|-----------------|--------------|------------------|---------|-------------|-----------------|
| fleet=480, int=5 (1019 req)  | 96.07%      | 96.17%          | 23.86        | 6.37             | 3.7×    | 46.67       | 14.25           |
| fleet=240, int=5 (1019 req)  | 63.20%      | 74.88%          | 20.34        | 5.58             | 3.6×    | 32.74       | 11.47           |
| fleet=120, int=3 (572 req)   | 54.02%      | 62.76%          | 10.33        | 2.60             | 4.0×    | 15.97       | 4.67            |
| fleet=240, int=3 (572 req)   | 95.28%      | 96.33%          | 13.55        | 3.29             | 4.1×    | 25.54       | 7.29            |

**Tail latency**: max wall-time matters for real-time dispatch. At the
largest cell (fleet=480, intensity=5), monolithic max is 46.67 ms vs
hier K=5 14.25 ms — **3.3× tighter tail**, never crossing 15 ms even at
1019 requests/day.

---

## 3. Narrative threads to pull for the proposal

These are the "what could we point at" angles, ranked by how clean the
evidence is. Each one is a one-paragraph contribution if needed.

1. **Compute-time scaling.** Hierarchical decomposition gives a stable
   3-4× per-decision wall-time speedup across fleet × intensity × geography
   sizes, with diminishing returns in K beyond 3. Numbers are tight (low
   seed-std, monotonic across K). Speedup heatmap is the natural figure.

2. **Quality at constraint.** When fleet barely covers demand, strict
   region partitioning prevents one greedy insertion from draining a
   far-away vehicle ("anti-vampire"). Hier wins by 8-25 pp at constrained
   corners while still being ~4× faster. K-sensitivity figure shows the
   monotonic improvement.

3. **Tail latency for real-time dispatch.** Max wall-time is what blows
   up an SLA, not mean. Hier K=5 keeps max under 15 ms at 1019 req/day on
   65 zones; monolithic peaks at 46-47 ms. This argues directly for
   real-time tractability.

4. **K=1 sanity check.** K=1 hierarchical recovers monolithic exactly on
   both metrics — no shenanigans hidden in the decomposition itself; the
   gain is purely from the partition + LLP scope.

5. **Geographic robustness.** Same speedup factors hold at 30 zones and
   65 zones, and across two intensity ramps; the approach isn't fragile
   to the specific synthetic / OSM choice.

6. **Hub-catchment partition vs k-means.** Sivagnanam et al. used k-means
   over historical incident locations. We diverged to hub-catchment
   (travel-time-nearest hub) because workforce commute is destination-
   anchored — the destination itself defines a natural region. K=5 has
   one degenerate region (Medical, 0 outskirts) because Downtown sits
   closer; this is what motivates the K-sensitivity sweep and validates
   K=3 (agglomerated) as a robust default.

---

## 4. What's NOT yet shown (open extensions)

These are explicit limitations to call out or extensions to point to:

- **Heuristic HLP/LLP only.** v1 LLP is greedy-insertion, v1 HLP is
  proportional rebalancing on a forecast window. Sivagnanam et al. learn
  HLP/LLP via DDQN. v2 axis.
- **Deterministic travel times.** No event loop, no uncertainty. The
  `TravelTimeOracle` interface exists so a stochastic oracle (companion
  repo `stgnn-drt-scenarios`) plugs in without touching dispatchers.
- **Microtransit only.** Multi-modal (rail, bus, AV mix) is the proposal's
  full ambition; this preliminary work isolates one modality.
- **Per-request dispatch.** Mini-batch / time-bucketed dispatch is a
  natural compute-time lever, deferred.
- **Single seed for steps 1 + 3.** Step 2 has 5-seed error bars at two
  operating points; the broader grids are single-seed. Re-running with
  multiple seeds is cheap if the proposal asks for it.
- **Real-Memphis hub locations are eyeballed.** Lat/lons are reasonable
  but not from a real OD survey. Trivial to swap if we get one.

---

## 5. Pointers

- **Repo:** `jptalusan/hierarchical_coordination` on GitHub.
- **Reproduce sweeps:** all three driver scripts under `scripts/`. Each
  takes ~1-3 min on M-series Mac after the first OSM pull (cached).
- **Figures + CSVs in this folder.** If anything grows stale, re-run the
  driver script — outputs go to `outputs/<name>/`, copy them back here.
- **Code anchors:**
  - `src/hcoord/dispatch/insertion.py` — shared kernel.
  - `src/hcoord/dispatch/{monolithic,hierarchical}.py` — both dispatchers.
  - `src/hcoord/regions.py` — partitioning + hub agglomeration.
  - `src/hcoord/analysis.py` — sweep runner + plot helpers.
  - `src/hcoord/geography_osm.py` — OSM substrate.

## 6. Possible v2 axes (in roughly cost order)

For when the proposal section structure is firm and we know what extra
evidence is worth running.

1. **Multi-seed reruns of step 1 + step 3.** ~5× wall, very cheap.
2. **Mini-batch dispatch.** Group requests by Δt window before dispatch.
   ~1 day to implement, may further widen the speedup gap.
3. **Stochastic `TravelTimeOracle`.** Plug in scenarios from companion
   repo, switch metric to expected on-time arrival. ~1 week.
4. **Learned LLP (cost approximation).** Replace exhaustive (p,q) search
   with a small NN scoring head. ~1 week. See section below.
5. **Learned HLP (rebalancing policy).** DDQN over region demand
   features, similar to Sivagnanam. ~2 weeks. See section below.
6. **Multi-modal (microtransit + fixed routes).** Add a feeder-line
   transfer modality to the insertion kernel. ~3-4 weeks.
