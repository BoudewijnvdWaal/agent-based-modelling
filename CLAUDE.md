# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
cd baseline_code
python run_me.py
```

No `requirements.txt` exists. Dependencies are managed via Conda. Required packages: `pandas`, `networkx`, `matplotlib`, `pygame`.

## Key Parameters (`baseline_code/run_me.py`, lines 29–59)

| Parameter | Default | Description |
|---|---|---|
| `simulation_duration_hours` | `6` | Simulation length (6 pseudo-hours = 360 ticks) |
| `GSE_COUNT` | `5` | Number of GSE vehicles |
| `GSE_SPEED` | `4.0` | Movement speed (units/minute) |
| `GSE_ELECTRIC` | `True` | Electric (15 min charge) vs gas (2 min charge) |
| `visualization` | `True` | Enable pygame window |
| `real_minutes_per_pseudo_hour` | `1.0` | Rendering speed multiplier |
| `status_print_interval` | `10` | Terminal table update frequency (ticks) |

## Architecture

This is an agent-based model of airport ground service operations at EHAM (Schiphol).

### Agents

**GSE (`GSE.py`) — active autonomous vehicles**
State machine: `available` → `taxiing` → `working` → `charging` / `needs_charging` → `at_cargo_pickup` → `at_cargo_dropoff`. Manages battery SoC (drains based on speed × distance), computes auction bids (`distance + battery_penalty`), and follows A* paths. Key thresholds: `low_soc_threshold=20%`, `critical_soc_threshold=10%`. Accepts `forbidden_nodes` on `plan_to_node()` for dynamic rerouting.

**Plane (`Plane.py`) — passive entities**
Spawned from Excel schedule. Progresses: `scheduled` → `awaiting_unload` → `awaiting_load` → `ready_to_depart` → `departed`. Turnaround time (departed − spawn) is the primary performance metric.

### Task Allocation

**`auction_system.py`** — Sequential single-item auction each tick. Each available GSE bids; lowest bid (distance + battery penalty) wins. Bid is ∞ if SoC too low or route unreachable.

### Pathfinding

**`single_agent_planner.py`** — A* over a NetworkX graph. Heuristics (exact Dijkstra distances) are precomputed once at startup. State space is time-expanded `(node, timestep)`. Accepts `forbidden_nodes` to skip blocked nodes; uses a depth limit (`len(nodes_dict)`) to prevent infinite loops when rerouting.

### Conflict Resolution — CBS (`cbs.py`)

Called every tick via `resolve_conflicts(gse_lst, nodes_dict, heuristics, t)` in the main loop. Four-stage algorithm:

1. **Spatio-temporal reservations** — each taxiing GSE reserves its current node and the next `RESERVATION_HORIZON=5` nodes with ETAs, plus the traversed edges (`frozenset({from, to}}`) to prevent "ghosting".
2. **Priority-based evasion** — lower-priority GSEs attempt to replan around reserved nodes/edges. If replanning fails, GSE is marked `waiting=True`.
3. **Cooperative rerouting** — higher-priority GSEs voluntarily reroute to unblock stuck agents.
4. **Deadlock breaker** — if ≥2 GSEs remain stuck, the lowest-ID one is forced to a free neighbor node to let others pass.

Replanning uses `_replan_around()`: temporarily removes blocked nodes from the graph, runs A*, then restores the graph.

### Main Simulation Loop (`run_me.py`, lines ~414–571)

Each tick (`dt=1.0` minute): spawn planes → complete services → despawn departed planes → update gate occupancy → auction tasks → send low-battery GSEs to charge → render → **resolve conflicts** → move GSEs (with physics substeps) → drain battery → release loads after unloading → transition service states → print status table.

### Data

All input from `baseline_code/Data/`:
- `nodes_EHAM.xlsx` — Node positions (x/y) and types (`gate`, `cargo`, `charging`)
- `edges_EHAM.xlsx` — Edges with distances
- `Plane_data.xlsx` — Aircraft schedule: spawn times, gate assignments, cargo origins/destinations

### Supporting Files

- `Fleet_manager.py` — Binary gate occupancy map (updated each tick from parked planes + GSEs)
- `visualization.py` — Pygame renderer: graph, GSE positions/headings/SoC, aircraft at gates
- `independent.py`, `prioritized.py` — Stub planners, not integrated
- `run logs/` — Written after each run (turnaround times, GSE energy/task reports)

## Code Notes

- Variable names and comments are mixed Dutch/English.
- Heuristics are precomputed once before the main loop using `calc_heuristics(nodes_dict, edges_dict)`.
- `GSE_RANDOM_SEED` controls random spawn locations; set to an integer for reproducible runs.
- The `rerouting` branch contains the active CBS conflict-resolution implementation in `cbs.py`; on `main`, `cbs.py` was a stub.
