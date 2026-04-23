# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
cd baseline_code
python run_me.py
```

For a batch of parallel runs:

```bash
cd baseline_code
python batch_run.py
```

No `requirements.txt` exists. Dependencies are managed via Conda. Required packages: `pandas`, `networkx`, `matplotlib`, `pygame`, `scipy`.

## Key Parameters (`baseline_code/run_me.py`, lines ~29–59)

| Parameter | Default | Description |
|---|---|---|
| `simulation_duration_hours` | `12` | Simulation length (12 pseudo-hours = 720 ticks) |
| `GSE_COUNT` | `7` | Number of GSE vehicles |
| `GSE_SPEED` | `4.0` | Movement speed (units/minute) |
| `GSE_ELECTRIC` | `False` | Electric (15 min charge, 0.5%/unit) vs gas (2 min charge) |
| `GSE_RANDOM_SEED` | `None` | Set to an integer for reproducible GSE spawn locations |
| `visualization` | `True` | Enable pygame window |
| `real_minutes_per_pseudo_hour` | `0.25` | Rendering speed (12 pseudo-hours → 3 real minutes) |
| `status_print_interval` | `10` | Terminal table update frequency (ticks) |

## Architecture

This is an agent-based model of airport ground service operations at EHAM (Schiphol).

### Agents

**GSE (`GSE.py`) — active autonomous vehicles**
State machine: `available` → `taxiing` → `working` → `charging` / `needs_charging` → `at_cargo_pickup` → `at_cargo_dropoff`. Battery consumption scales with `speed` so energy per unit distance is constant regardless of speed setting. Key SoC thresholds: `low_soc_threshold=20%` (stops bidding, transitions to `needs_charging`), `critical_soc_threshold=10%` (forces depot return even mid-route). Charging is a fixed-duration event (electric: 15 min, gas: 2 min) to 100%. After finishing a task, `finish_working()` auto-routes to depot if SoC < 20%.

**Plane (`Plane.py`) — passive entities**
Spawned from Excel schedule. Progresses: `scheduled` → `awaiting_unload` → `awaiting_load` → `ready_to_depart` → `departed`. Turnaround time (departed − spawn) is the primary performance metric.

### Task Allocation

**`auction_system.py`** — Sequential single-item auction each tick. Bid formula: `distance + (100 − SoC) × 0.2`. Bid is ∞ if GSE is unavailable, SoC too low, or route unreachable. The battery penalty (× 0.2) favors fuller GSEs without overriding proximity. Bid covers only the task route, not return to depot.

### Pathfinding

**`single_agent_planner.py`** — A* over a NetworkX graph. Heuristics (exact Dijkstra distances) are precomputed once at startup via `calc_heuristics()`. State space is time-expanded `(node, timestep)`. Accepts `forbidden_nodes` to skip blocked nodes; depth-limited to `len(nodes_dict)` to prevent infinite loops.

### Conflict Resolution — CBS (`cbs.py`)

Called every tick via `resolve_conflicts(gse_lst, nodes_dict, heuristics, t)`. Priority rule: **lower GSE ID = higher priority** (a GSE only yields to lower-ID GSEs). `RESERVATION_HORIZON = 3` look-ahead steps.

Four-stage algorithm:
1. **Spatio-temporal reservations** — each taxiing GSE reserves its current node and next 3 path nodes with ETAs, plus traversed edges as `frozenset({from, to})` to prevent "ghosting".
2. **Priority-based evasion** — higher-ID (lower-priority) GSEs detect conflicts with lower-ID paths and call `_replan_around()`: temporarily removes blocked nodes from the graph, runs A*, restores the graph. Failed replans mark the GSE `waiting=True`.
3. **Cooperative rerouting** — lower-priority bystanders reroute to free a corridor for stuck higher-priority GSEs.
4. **Deadlock breaker** — if ≥2 GSEs remain stuck, the lowest-priority (highest-ID) one is forced to a free neighbor node. After any successful replan, reservations are updated immediately so subsequent GSEs see the corrected picture.

### Main Simulation Loop (`run_me.py`)

Each tick (`dt=1.0` minute):
1. Spawn planes from schedule
2. Complete working GSEs → transition service states
3. Despawn departed planes
4. Update gate occupancy (`Fleet_manager`)
5. Auction service tasks
6. Route low-battery GSEs to nearest charging node (factoring congestion)
7. Render frame (if visualization, with movement sub-stepping for smooth animation)
8. **Resolve conflicts** (CBS) ← runs after render, before movement
9. Move GSEs
10. Drain battery
11. Release loads post-unload; run second auction round for cargo-loaded GSEs
12. Print status table

### Batch Runner & Analysis (`baseline_code/`)

**`batch_run.py`** — Parallel runs via `ProcessPoolExecutor` (default: CPU count − 1 workers). Each seed uses `build_random_gse_spawn_config(seed=seed)` for identical GSE spawn configs across electric/gas comparisons. Output per batch folder:
- `batch_results.csv` — per-seed summary (mean TAT, n_completed)
- `planes_summary.csv` / `gse_summary.csv` — per-plane and per-GSE metrics
- `latest_batch.txt` — pointer consumed by analysis scripts

**`find_n_method_1.py`** — Sequential stopping rule: runs paired electric/gas seeds one-at-a-time, stops when running CI width < L (default 5 min).

**`find_n_method_2.py`** — Confidence interval approach for minimum sample size determination.

**`experiment_1_analysis.py`** — Two-tailed paired t-test on mean TAT difference (gas − electric). Reports point estimate, 95% CI, Cohen's d, t-statistic, and p-value. Reads from `latest_batch.txt`.

### Data

All input from `baseline_code/Data/`:
- `nodes_EHAM.xlsx` — Node positions (x/y) and types (`gate`, `cargo`, `charging`)
- `edges_EHAM.xlsx` — Edges with distances
- `Planes.xlsx` — Aircraft schedule: spawn times, gate assignments, cargo origins/destinations

### Supporting Files

- `Fleet_manager.py` — Binary gate occupancy map (updated each tick from parked planes + GSEs)
- `visualization.py` — Pygame renderer: graph, GSE positions/headings/SoC, aircraft at gates
- `independent.py`, `prioritized.py` — Stub planners, not integrated
- `run logs/` — Written after each run (turnaround times, GSE energy/task reports)

## Code Notes

- Variable names and comments are mixed Dutch/English.
- Heuristics are precomputed once before the main loop using `calc_heuristics(nodes_dict, edges_dict)`.
- The `rerouting` branch is the active development branch containing the full CBS implementation. On `main`, `cbs.py` was a stub.
