# Whitepaper: Hierarchical Coordination for Thrust 2

## Files

- `thrust2_hierarchical_coordination.tex` — 1-2 page whitepaper aligned to NSF
  TTP-T Thrust 2 (RQ4/RQ5, Barriers 1-3, M2/M4 milestones). Targets the
  preliminary-evidence section of the proposal write-up.
- `../learned_llp/wall_vs_fleet.png` — figure referenced from the .tex.
- `../learned_llp/results_stacked.csv`, `aggregated_stacked.csv` — raw and
  per-cell aggregated numbers behind Table 1.

## Compile

```
cd write-up/whitepaper
pdflatex thrust2_hierarchical_coordination.tex
```

(No bibliography file — citations are inlined in prose. If you add
references, drop in a `.bib` and add `\bibliographystyle/\bibliography`.)

## Where the numbers come from

Every quantitative claim in the .tex maps to a committed sweep:

| claim                                | script                                  | output                                       |
|--------------------------------------|-----------------------------------------|----------------------------------------------|
| 3-4× hierarchical-only speedup       | `scripts/scaleup_sweep.py`              | `write-up/scaleup/results.csv`               |
| K-sensitivity / +12.8pp at fleet=60  | `scripts/ksensitivity_sweep.py`         | `write-up/ksensitivity/aggregated.csv`       |
| 65-zone scale-up                     | `scripts/largescale_sweep.py`           | `write-up/largescale/results.csv`            |
| LLP feasibility 98.9%, MAE 0.52 min  | `scripts/train_scorer.py`               | `write-up/learned_llp/scorer.eval.json`      |
| Filter / stacked / hier+filter table | `scripts/learned_llp_stacked_sweep.py`  | `write-up/learned_llp/results_stacked.csv`   |
| HLP spot-check (parked as v2)        | `scripts/hlp_spot_check.py`             | `write-up/learned_hlp/spot_check_findings.md`|

## Updating the figure

If sweep numbers change, regenerate `wall_vs_fleet.png`:

```
uv run python scripts/learned_llp_stacked_sweep.py
cp outputs/learned_llp_stacked/wall_vs_fleet.png write-up/learned_llp/
```

The .tex references it via relative path `../learned_llp/wall_vs_fleet.png`.

## Structure cross-walk to the proposal

| .tex section                                | maps to proposal section |
|---------------------------------------------|--------------------------|
| §1 Motivation and Research Question         | §2.3.2, RQ 5             |
| §2 Approach                                 | §3.2 Barrier 2 (RL high / MPC low) |
| §3 Preliminary Results                      | (new — preliminary evidence) |
| §4 Composition with Uncertainty Modeling    | §3.2 Barrier 1           |
| §5 Position in the Multi-Modal Picture      | §1 Overview, §3.1        |
| §6 Digital-Twin Evaluation Plan             | §3.2 Barrier 3, §3.4 Phase I |
| §7 Status and Roadmap                       | §3.5 Project Timeline (M2, M4) |
