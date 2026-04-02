# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
cd baseline_code
python run_me.py
```

No `requirements.txt` exists. Dependencies are managed via Conda. Required packages: `pandas`, `networkx`, `matplotlib`, `pygame`.

## Key Parameters (baseline_code/run_me.py, lines 23–48)

| Parameter | Default | Description |
|---|---|---|
| `simulation_duration_hours` | `12` | Simulation length (12 pseudo-hours = 720 ticks) |
| `GSE_COUNT` | `5` | Number of GSE vehicles |
| `GSE_SPEED` | `4.0` | Movement speed |
| `DEPOT_NODE` | `2` | Charging station node ID |
| `visualization` | `True` | Enable pygame window |
| `planner` | `"Independent"` | Path planning strategy |

## Architecture

This is an agent-based model of airport ground service operations at EHAM (Schiphol).

### Agents

**GSE (Ground Support Equipment) — `GSE.py`**
Active agents (autonomous vehicles) that service aircraft. They have a state machine (`available` → `taxiing` → `working` → `charging` / `needs_charging` → `at_cargo_pickup` → `at_cargo_dropoff`), manage battery SoC, compute auction bids (distance + battery penalty), and follow A* paths.

**Plane — `Plane.py`**
Passive entities spawned from an Excel schedule. Each aircraft progresses through: `scheduled` → `awaiting_unload` → `awaiting_load` → `ready_to_depart` → `departed`. Cargo has explicit origin/destination nodes. Turnaround time is the primary performance metric.

### Task Allocation

**`auction_system.py`** — Sequential single-item auction. Each available GSE bids; lowest bid (distance + battery penalty) wins the task.

### Pathfinding

**`single_agent_planner.py`** — A* with Dijkstra-precomputed heuristics over a NetworkX graph. Heuristics are precomputed once at startup in `run_me.py`.

### Data

All input comes from `baseline_code/Data/`:
- `nodes_EHAM.xlsx` — Node positions (x/y) and types (`gate`, `cargo`, `charging`)
- `edges_EHAM.xlsx` — Edges with distances between nodes
- `Plane_data.xlsx` — Aircraft schedule: spawn times, gate assignments, cargo origins/destinations

### Visualization

**`visualization.py`** — Pygame renderer. Shows the airport graph, aircraft at gates, GSE positions/headings, and battery states in real time.

### Supporting Files

- `Fleet_manager.py` — Tracks gate occupancy to prevent conflicts
- `cbs.py`, `prioritized.py`, `independent.py` — Planner stubs (not yet fully implemented)
- `simulation_outputs/` — Written after each run (turnaround times, GSE reports)

## Code Notes

- Comments and variable names are mixed Dutch/English (airport domain context).
- The main simulation loop is in `run_me.py` (~lines 286–503): spawn planes → auction tasks → move GSEs → handle service completions → manage charging → render.
- `cbs.py`, `prioritized.py`, `independent.py` are stub files for multi-agent path planning strategies that are not yet integrated.
