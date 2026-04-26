# Agent-Based Modelling: Airport Ground Operations

This project simulates airport ground handling (EHAM) using agent-based modelling.
Aircraft are spawned from a schedule, and GSE vehicles (Ground Support Equipment)
handle unload/load tasks through an auction mechanism with route conflict resolution.

## Project Structure

- `baseline_code/`: core simulation code
- `baseline_code/run_me.py`: main script for a single simulation run
- `baseline_code/batch_run.py`: parallel batch runs and gas vs electric comparison
- `baseline_code/cbs.py`: conflict resolution with reservations and rerouting
- `baseline_code/single_agent_planner.py`: A* path planning and heuristics
- `baseline_code/GSE.py`: GSE agent behavior and state machine
- `baseline_code/Plane.py`: aircraft behavior and lifecycle
- `baseline_code/auction_system.py`: task assignment via bidding
- `baseline_code/visualization.py`: pygame visualization
- `baseline_code/experiment_1_analysis.py`: statistical analysis of batch results
- `baseline_code/find_n_method_1.py`: sequential stopping rule for required n
- `baseline_code/find_n_method_2.py`: confidence-interval method for required n
- `baseline_code/verification_checks.py`: standalone verification checks (manual script)
- `run logs/`: output files from runs and batches
- `sensitivity_analysis/`: scripts and outputs for sensitivity analysis

## Quick Start

Working directory:

```powershell
cd baseline_code
```

Run a single simulation:

```powershell
python run_me.py
```

Run batch experiments:

```powershell
python batch_run.py
```

## Key Parameters

In `baseline_code/run_me.py`, the parameter section includes:

- `simulation_duration_hours`
- `GSE_COUNT`
- `GSE_SPEED`
- `GSE_ELECTRIC`
- `GSE_RANDOM_SEED`
- `AUCTION_ALPHA` and `AUCTION_BETA`
- `visualization`

## High-Level Simulation Flow

Each tick follows roughly this sequence:

1. Spawn new aircraft.
2. Complete running services and update statuses.
3. Despawn departed aircraft.
4. Update gate occupancy.
5. Auction and assign tasks.
6. Route low-SoC GSEs to charging stations.
7. Resolve route conflicts (CBS).
8. Move vehicles and update battery.
9. Optionally run a second auction round (after load release).

## Data

Input data is stored in `baseline_code/Data/`:

- `nodes_EHAM.xlsx`
- `edges_EHAM.xlsx`
- `airport_schedule_24h.xlsx`

## Dependency Note

There is no `requirements.txt` in this repository.
Use a Python/Conda environment with at least:

- pandas
- networkx
- matplotlib
- pygame
- scipy

## Inactive or Standalone Parts

These files are present but not part of the standard runtime flow in `run_me.py`:

- `baseline_code/independent.py` (stub)
- `baseline_code/prioritized.py` (stub)
- `baseline_code/verification_checks.py` (standalone manual checks)

They were intentionally kept and not removed.
