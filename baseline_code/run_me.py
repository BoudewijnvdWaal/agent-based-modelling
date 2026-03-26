"""
Run-me.py is the main file of the simulation. Run this file to run the simulation.
"""

from pathlib import Path
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as timer
import pygame as pg
from single_agent_planner import calc_heuristics
from visualization import map_initialization, map_running
from GSE import GSE
from Plane import load_plane_schedule, build_plane_schedule_lookup, spawn_planes
from Fleet_manager import Fleet_manager
from auction_system import AuctionSystem

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# SIMULATION PARAMETERS  (pas hier aan)
# =============================================================================
nodes_file = "Data/nodes_EHAM.xlsx"
edges_file = "Data/edges_EHAM.xlsx"
plane_data_file = "Data/Plane_data.xlsx"

simulation_duration_hours = 12
simulation_time = simulation_duration_hours * 60  # 12 pseudo-hours = 720 simulated minutes
planner = "Independent"

# Hoe lang een vliegtuig aan de gate staat (zelfde tijdseenheid als t)
gate_turnaround_time = 3.0

# --- [NIEUW] GSE configuratie ---
GSE_COUNT = 5
GSE_RANDOM_SEED = None
DEPOT_NODE = 2   # node_id van het laadstation/depot; pas aan naar jouw layout

# Visualisatie
plot_graph         = False
visualization      = True
step_duration_seconds = 0.25
render_every_n_steps = 1


# =============================================================================
# FUNCTIE-DEFINITIES  (ongewijzigd t.o.v. origineel, behalve spawn_aircrafts)
# =============================================================================

def import_layout(nodes_file, edges_file):
    """
    Imports layout information from xlsx files and converts this into dictionaries.
    INPUT:
        - nodes_file = xlsx file with node input data
        - edges_file = xlsx file with edge input data
    RETURNS:
        - nodes_dict = dictionary with nodes and node properties
        - edges_dict = dictionary with edges and edge properties
        - start_and_goal_locations = dictionary with node ids for arrival runways,
          departure runways and gates
    """
    gates_xy      = []
    cargoep_xy    = []
    chargingrr_xy = []

    def resolve_path(path_like):
        path = Path(path_like).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.append(BASE_DIR / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find layout file '{path_like}'. "
            f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
        )

    nodes_path = resolve_path(nodes_file)
    edges_path = resolve_path(edges_file)

    df_nodes = pd.read_excel(nodes_path)
    df_edges = pd.read_excel(edges_path)

    nodes_dict = {}
    for i, row in df_nodes.iterrows():
        node_properties = {
            "id":        row["id"],
            "x_pos":     row["x_pos"],
            "y_pos":     row["y_pos"],
            "xy_pos":    (row["x_pos"], row["y_pos"]),
            "type":      row["type"],
            "neighbors": set()
        }
        node_id = row["id"]
        nodes_dict[node_id] = node_properties

        if row["type"] == "cargo":
            cargoep_xy.append((row["x_pos"], row["y_pos"]))
        elif row["type"] == "charging":
            chargingrr_xy.append((row["x_pos"], row["y_pos"]))
        elif row["type"] == "gate":
            gates_xy.append((row["x_pos"], row["y_pos"]))

    start_and_goal_locations = {
        "gates":   gates_xy,
        "dep_rwy": cargoep_xy,
        "arr_rwy": chargingrr_xy
    }

    edges_dict = {}
    for i, row in df_edges.iterrows():
        edge_id       = (row["from"], row["to"])
        from_node     = edge_id[0]
        to_node       = edge_id[1]
        start_end_pos = (nodes_dict[from_node]["xy_pos"], nodes_dict[to_node]["xy_pos"])
        edge_properties = {
            "id":            edge_id,
            "from":          row["from"],
            "to":            row["to"],
            "length":        row["length"],
            "weight":        row["length"],
            "start_end_pos": start_end_pos
        }
        edges_dict[edge_id] = edge_properties

    for edge in edges_dict:
        nodes_dict[edge[0]]["neighbors"].add(edge[1])

    return nodes_dict, edges_dict, start_and_goal_locations


def create_graph(nodes_dict, edges_dict, plot_graph=True):
    """
    Creates networkX graph based on nodes and edges and optionally plots it.
    """
    graph = nx.DiGraph()

    for node in nodes_dict:
        graph.add_node(node,
                       node_id   = nodes_dict[node]["id"],
                       xy_pos    = nodes_dict[node]["xy_pos"],
                       node_type = nodes_dict[node]["type"])

    for edge in edges_dict:
        graph.add_edge(edge[0], edge[1],
                       edge_id   = edge,
                       from_node = edges_dict[edge]["from"],
                       to_node   = edges_dict[edge]["to"],
                       weight    = edges_dict[edge]["length"])

    if plot_graph:
        plt.figure()
        node_locations = nx.get_node_attributes(graph, 'xy_pos')
        nx.draw(graph, node_locations, with_labels=True, node_size=100, font_size=10)

    return graph


def build_random_gse_spawn_config(nodes_dict, gse_count, seed=None):
    """
    Generate (gse_id, start_node) tuples for a randomly placed GSE fleet.
    GSEs spawn on non-gate nodes to avoid occupying aircraft gates at t=0.
    """
    candidate_nodes = [
        node_id
        for node_id, node_props in nodes_dict.items()
        if node_props.get("type") != "gate"
    ]
    if not candidate_nodes:
        raise ValueError("No non-gate nodes available for random GSE spawning.")

    rng = random.Random(seed)
    if gse_count <= len(candidate_nodes):
        spawn_nodes = rng.sample(candidate_nodes, gse_count)
    else:
        spawn_nodes = [rng.choice(candidate_nodes) for _ in range(gse_count)]

    return [(gse_id, start_node) for gse_id, start_node in enumerate(spawn_nodes, start=1)]

# =============================================================================
# INITIALISATIE
# =============================================================================
nodes_dict, edges_dict, start_and_goal_locations = import_layout(nodes_file, edges_file)
graph      = create_graph(nodes_dict, edges_dict, plot_graph)
heuristics = calc_heuristics(graph, nodes_dict)

# --- Geplande vliegtuigen uit Plane_data.xlsx ---
plane_schedule = load_plane_schedule(
    plane_data_file,
    nodes_dict,
    gate_turnaround_time,
    base_dir=BASE_DIR,
)
plane_schedule_lookup = build_plane_schedule_lookup(plane_schedule)
active_planes = []
gse_spawn_config = build_random_gse_spawn_config(
    nodes_dict,
    GSE_COUNT,
    seed=GSE_RANDOM_SEED,
)

# --- [NIEUW] Maak GSE-vloot aan ---
gse_lst = []
for gse_id, start_node in gse_spawn_config:
    gse = GSE(gse_id=gse_id, start_node=start_node,
              nodes_dict=nodes_dict, depot_node=DEPOT_NODE)
    gse_lst.append(gse)
print(f"[Init] {len(gse_lst)} GSEs aangemaakt met spawn nodes: {gse_spawn_config}")

# --- [NIEUW] Fleet Manager en Auction System ---
fleet_manager  = Fleet_manager(nodes_dict)
auction_system = AuctionSystem(gse_lst)

# Bijhouden welke vliegtuigen al een GSE toegewezen hebben gekregen
# zodat we niet bij elke tijdstap opnieuw een veiling houden voor hetzelfde vliegtuig.
already_auctioned = set()

if visualization:
    map_properties = map_initialization(nodes_dict, edges_dict)


# =============================================================================
# SIMULATIELUS
# =============================================================================
running       = True
escape_pressed = False
time_end      = simulation_time
dt            = 1.0   # 1 simulatiestap = 1 pseudo-minuut
t             = 0
step_count    = 0

print("Simulation Started")
while running:
    t = round(t, 2)

    # --- Stopconditie ---
    if t >= time_end or escape_pressed:
        running = False
        pg.quit()
        print("Simulation Stopped")
        break

    # -------------------------------------------------------------------------
    # Spawn / verwijder geplande vliegtuigen uit Plane_data.xlsx
    # -------------------------------------------------------------------------
    new_planes = spawn_planes(t, plane_schedule_lookup)
    if new_planes:
        active_planes.extend(new_planes)

    # Verwijder vliegtuigen waarvan de turnaround voorbij is EN waarvan de GSE is aangekomen.
    # Een plane blijft zichtbaar totdat de toegewezen GSE status "working" heeft op die gate,
    # zodat vliegtuigen niet verdwijnen voordat ze bediend zijn.
    working_gate_nodes = {
        gse.current_node
        for gse in gse_lst
        if gse.status == "working"
    }
    despawned_ids = {
        plane.id
        for plane in active_planes
        if plane.ready_to_despawn(t, working_gate_nodes)
    }
    if despawned_ids:
        for plane in active_planes:
            if plane.id in despawned_ids:
                plane.status = "departed"
                already_auctioned.discard(plane.id)
        active_planes = [plane for plane in active_planes if plane.id not in despawned_ids]

    # -------------------------------------------------------------------------
    # [NIEUW] Fleet Manager: update gate-bezettingskaart
    # -------------------------------------------------------------------------
    fleet_manager.update_gate_status(active_planes, aircraft_lst=gse_lst, t=t)

    # -------------------------------------------------------------------------
    # [NIEUW] Auction: wijs GSEs toe aan gates die nog niet bediend worden
    # -------------------------------------------------------------------------
    unassigned_planes = [
        plane
        for plane in active_planes
        if not plane.serviced and plane.id not in already_auctioned
    ]

    if unassigned_planes:
        assignments = auction_system.allocate_tasks(unassigned_planes, heuristics)
        for gse, plane in assignments:
            already_auctioned.add(plane.id)
            plane.serviced = True
            gse.plan_to_gate(plane.node_id, nodes_dict, heuristics, t)

    # -------------------------------------------------------------------------
    # [NIEUW] GSEs die 'needs_charging' zijn sturen naar het depot
    # -------------------------------------------------------------------------
    for gse in gse_lst:
        if gse.status == "needs_charging":
            gse.go_charge(nodes_dict, heuristics, t)

    # -------------------------------------------------------------------------
    # Visualisatie
    # -------------------------------------------------------------------------
    if visualization and step_count % render_every_n_steps == 0:
        # Toon alle GSEs, ook wanneer ze stilstaan op hun spawnlocatie.
        current_states = {}
        for gse in gse_lst:
            current_states[gse.id] = {
                "ac_id":   f"GSE {gse.id}",
                "xy_pos":  gse.position,
                "heading": gse.heading
            }

        gate_states = {
            plane.id: plane.to_gate_state()
            for plane in active_planes
        }
        escape_pressed = map_running(map_properties, current_states, gate_states, t)
    if step_duration_seconds > 0:
        timer.sleep(step_duration_seconds)

    # -------------------------------------------------------------------------
    # [NIEUW] Beweeg GSEs + update SoC
    # -------------------------------------------------------------------------
    for gse in gse_lst:
        if gse.status == "taxiing":
            gse.move(dt, t)
        gse.update_soc(dt)

    # -------------------------------------------------------------------------
    # [NIEUW] GSEs die bij het depot zijn aangekomen: zet op 'charging'
    # (move() zet status al op 'charging' via _on_goal_reached, dit is een
    #  extra vangnet voor het geval de logica afwijkt)
    # -------------------------------------------------------------------------
    for gse in gse_lst:
        if gse.status == "arrived" and gse.current_node == DEPOT_NODE:
            gse.status = "charging"

    t = t + dt
    step_count += 1


# =============================================================================
# ANALYSE VAN OUTPUTDATA
# =============================================================================
# Voeg hier analyse toe, bijvoorbeeld:
#   - Gemiddelde SoC over de tijd
#   - Aantal voltooide taken per GSE
#   - Conflicten of wachttijden
print("\n--- Eindrapport GSEs ---")
for gse in gse_lst:
    print(gse)
