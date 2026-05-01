# hcoord — hierarchical decomposition for multi-modal DVRP (preliminary)

Preliminary work for NSF TTP-T Thrust 2: showing that hierarchical decomposition
makes real-time multi-modal microtransit dispatch tractable at fleet scale.

## What this is

A small simulation environment + dispatcher comparison:
- Synthetic Memphis-outskirts → Memphis-hubs geography (30 zones, 5 hubs).
- Capacitated microtransit fleet, shift-synchronized demand.
- Two dispatchers sharing the same insertion primitive:
  - `Monolithic`: full-corridor insertion (baseline).
  - `Hierarchical`: HLP rebalances vehicle counts across hub-catchment regions;
    one LLP per region inserts requests into in-region routes.
- Headline figures: compute time vs. fleet/request rate, on-time arrival rate,
  K-region sensitivity.

## What this is not

- Not a full multi-modal system yet — microtransit only.
- Not learned: HLP and LLP are heuristic in v1. Learned variants are a v2 axis.
- Not a real Memphis road graph — synthetic placement around hub centers.
- Not uncertainty-aware: travel times are deterministic. Uncertainty plugs in
  via the `TravelTimeOracle` interface (companion repo `stgnn-drt-scenarios`).

## Layout

```
src/hcoord/
  geography.py   # Zone, Network, build_memphis_outskirts()
  travel.py      # TravelTimeOracle (cached all-pairs shortest paths)
  ...
tests/
```

## Running tests

```
uv sync --all-extras
uv run pytest
```

## Real-OSM Memphis substrate

Optional `[osm]` extra adds `osmnx` + `scikit-learn`. First call pulls the
Memphis drive network from OSM Overpass (~30 s) and caches the graph plus
the built `Network` to `data/osm_cache/`. Subsequent calls return instantly.

```
uv sync --extra osm
uv run python -m hcoord network=memphis_osm dispatcher=hierarchical n_regions=3
```
