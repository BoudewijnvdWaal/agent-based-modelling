# Verification Report
Generated: 2026-04-23T11:02:41

## Checks
- PASS | A* manual path | actual=[1, 2, 6, 7, 5], expected=[1, 2, 6, 7, 5], length=6
- PASS | Heuristics | nodes=8, pairs=64
- PASS | Bidding (position + SoC) | winner=1, position_ok=True, soc_effect_ok=True, low_soc_ok=True
- PASS | Min/max | soc=100.0, status=available
- PASS | Constraints | {'gate_ok': True, 'cargo_ok': True, 'charge_ok': True}

## A* example
- Actual path: [1, 2, 6, 7, 5]
- Expected path: [1, 2, 6, 7, 5]
- Plot: C:\Users\phili\Documents\TU Delft\MSc\AE - Agent_based_modelling\agent-based-modelling\baseline_code\Figures\astar_simple.png

## Bidding graph
- Plot: C:\Users\phili\Documents\TU Delft\MSc\AE - Agent_based_modelling\agent-based-modelling\baseline_code\Figures\bidding_graph.png
- Bidding processes plot: C:\Users\phili\Documents\TU Delft\MSc\AE - Agent_based_modelling\agent-based-modelling\baseline_code\Figures\bidding_processes.png

## Bidding
- Winner: GSE 1
- Position check: True
- SoC effect check: True (high=7.0, low=19.0)
- Low SoC block check: True (bid=inf)
- Bid rows: 2
- Log file: C:\Users\phili\Documents\TU Delft\MSc\AE - Agent_based_modelling\agent-based-modelling\baseline_code\Data\verification_logs\verification_log.md

### Bid details
- GSE 1: bid=8.0, soc=90.0, start_node=3
- GSE 2: bid=12.0, soc=60.0, start_node=7
