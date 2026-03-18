"""
Run-me.py is the main file of the simulation. Run this file to run the simulation.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as timer
import pygame as pg
from single_agent_planner import calc_heuristics
from visualization import map_initialization, map_running
from GSE import GSE
from Fleet_manager import Fleet_manager
from auction_system import AuctionSystem

# =============================================================================
# SIMULATION PARAMETERS  (pas hier aan)
# =============================================================================
nodes_file = "Data/nodes_EHAM.xlsx"
edges_file = "Data/edges_EHAM.xlsx"

simulation_time = 1000
planner = "Independent"

# Vliegtuigen rijden NIET meer over de taxibaan — ze spawnen direct bij een gate
# als statisch gate_plane object. GSEs zijn de enige bewegende agents.
# Laat deze lijst leeg, of verwijder hem helemaal.
spawn_schedule = []

# Statische vliegtuigen bij gates: (spawn_time, gate_node_id)
gate_plane_schedule = [
    (0.5, 7),
    (2.0, 9),
    (4.0, 14),
    (6.0, 17),
]

# Hoe lang een vliegtuig aan de gate staat (zelfde tijdseenheid als t)
gate_turnaround_time = 3.0

# --- [NIEUW] GSE configuratie ---
# Elke tuple: (gse_id, start_node)
# start_node moet een 'charging'-type node zijn (= depot).
# Pas de node-ids aan zodra je weet welke nodes 'charging' zijn in jouw layout.
GSE_SPAWN_CONFIG = [
    (1, 2),   # GSE 1 start op node 2
    (2, 2),   # GSE 2 start op node 2
    (3, 2),   # GSE 3 start op node 2
]
DEPOT_NODE = 2   # node_id van het laadstation/depot; pas aan naar jouw layout

# Visualisatie
plot_graph         = False
visualization      = True
visualization_speed = 0.1


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

    df_nodes = pd.read_excel(os.getcwd() + "/" + nodes_file)
    df_edges = pd.read_excel(os.getcwd() + "/" + edges_file)

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


def spawn_gate_planes(t, nodes_dict, schedule, turnaround_time, next_id_ref):
    """
    Create static gate planes whose spawn time matches the current timestep.
    Each plane remains visible until spawn_time + turnaround_time.
    """
    new_gate_planes = []
    for spawn_time, gate_node_id in schedule:
        if abs(spawn_time - t) < 1e-9:
            if gate_node_id not in nodes_dict:
                raise ValueError(
                    f"Gate node {gate_node_id} not found in nodes_dict; "
                    "update gate_plane_schedule."
                )
            position = nodes_dict[gate_node_id]["xy_pos"]
            new_gate_planes.append({
                "id":          next_id_ref[0],
                "node_id":     gate_node_id,
                "xy_pos":      position,
                "despawn_time": spawn_time + turnaround_time,
                "serviced":    False   # [NIEUW] bijhouden of er al een GSE is toegewezen
            })
            next_id_ref[0] += 1
    return new_gate_planes


# =============================================================================
# INITIALISATIE
# =============================================================================
nodes_dict, edges_dict, start_and_goal_locations = import_layout(nodes_file, edges_file)
graph      = create_graph(nodes_dict, edges_dict, plot_graph)
heuristics = calc_heuristics(graph, nodes_dict)

# --- Statische gate-vliegtuigen ---
gate_planes        = []
gate_plane_next_id = [1]

# --- [NIEUW] Maak GSE-vloot aan ---
gse_lst = []
for gse_id, start_node in GSE_SPAWN_CONFIG:
    gse = GSE(gse_id=gse_id, start_node=start_node,
              nodes_dict=nodes_dict, depot_node=DEPOT_NODE)
    gse_lst.append(gse)
print(f"[Init] {len(gse_lst)} GSEs aangemaakt: {gse_lst}")

# --- [NIEUW] Fleet Manager en Auction System ---
fleet_manager  = Fleet_manager(nodes_dict)
auction_system = AuctionSystem(gse_lst)

# Bijhouden welke gate_plane node_ids al een GSE toegewezen hebben gekregen
# zodat we niet bij elke tijdstap opnieuw een veiling houden voor dezelfde gate.
already_auctioned = set()

if visualization:
    map_properties = map_initialization(nodes_dict, edges_dict)


# =============================================================================
# SIMULATIELUS
# =============================================================================
running       = True
escape_pressed = False
time_end      = simulation_time
dt            = 0.1   # moet een factor van 0.5 zijn
t             = 0

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
    # Spawn / verwijder statische gate-vliegtuigen
    # -------------------------------------------------------------------------
    new_gate_planes = spawn_gate_planes(
        t, nodes_dict, gate_plane_schedule, gate_turnaround_time, gate_plane_next_id
    )
    if new_gate_planes:
        gate_planes.extend(new_gate_planes)

    # Verwijder vliegtuigen waarvan de turnaround voorbij is EN waarvan de GSE is aangekomen.
    # Een gate-plane blijft zichtbaar totdat de toegewezen GSE status "working" heeft op die gate,
    # zodat vliegtuigen niet verdwijnen voordat ze bediend zijn.
    def gse_has_arrived(gate_node_id):
        """Geeft True als een GSE op deze gate staat met status 'working'."""
        return any(
            gse.status == "working" and gse.current_node == gate_node_id
            for gse in gse_lst
        )

    despawned = [
        gp for gp in gate_planes
        if t >= gp["despawn_time"] - 1e-9 and (not gp["serviced"] or gse_has_arrived(gp["node_id"]))
    ]
    for gp in despawned:
        already_auctioned.discard(gp["node_id"])
    gate_planes = [gp for gp in gate_planes if gp not in despawned]

    # -------------------------------------------------------------------------
    # [NIEUW] Fleet Manager: update gate-bezettingskaart
    # -------------------------------------------------------------------------
    fleet_manager.update_gate_status(gate_planes, aircraft_lst=gse_lst, t=t)

    # -------------------------------------------------------------------------
    # [NIEUW] Auction: wijs GSEs toe aan gates die nog niet bediend worden
    # -------------------------------------------------------------------------
    unassigned_tasks = [
        gp["node_id"]
        for gp in gate_planes
        if not gp["serviced"] and gp["node_id"] not in already_auctioned
    ]

    if unassigned_tasks:
        assignments = auction_system.allocate_tasks(unassigned_tasks, heuristics)
        for gse, gate_node_id in assignments:
            already_auctioned.add(gate_node_id)
            # Markeer het gate-vliegtuig als bediend
            for gp in gate_planes:
                if gp["node_id"] == gate_node_id:
                    gp["serviced"] = True
            # Plan het pad van de winnende GSE naar de gate
            gse.plan_to_gate(gate_node_id, nodes_dict, heuristics, t)

    # -------------------------------------------------------------------------
    # [NIEUW] GSEs die 'needs_charging' zijn sturen naar het depot
    # -------------------------------------------------------------------------
    for gse in gse_lst:
        if gse.status == "needs_charging":
            gse.go_charge(nodes_dict, edges_dict, heuristics, t)

    # -------------------------------------------------------------------------
    # Visualisatie
    # -------------------------------------------------------------------------
    if visualization:
        # Alleen GSEs worden gevisualiseerd als bewegende agents
        current_states = {}
        for gse in gse_lst:
            if gse.status == "taxiing":
                current_states[gse.id] = {
                    "ac_id":   f"GSE {gse.id}",
                    "xy_pos":  gse.position,
                    "heading": gse.heading
                }

        gate_states = {
            gp["id"]: {"id": gp["id"], "node_id": gp["node_id"], "xy_pos": gp["xy_pos"]}
            for gp in gate_planes
        }
        escape_pressed = map_running(map_properties, current_states, gate_states, t)
        timer.sleep(visualization_speed)

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