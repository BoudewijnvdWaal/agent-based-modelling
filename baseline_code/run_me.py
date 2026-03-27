"""
Run-me.py is the main file of the simulation. Run this file to run the simulation.
"""

from pathlib import Path
import math
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

# Service- en vertrektijden van vliegtuigen, in pseudo-minuten
unloading_duration_minutes = 15
loading_duration_minutes = 15
departure_delay_minutes = 5

# --- [NIEUW] GSE configuratie ---
GSE_COUNT = 5
GSE_RANDOM_SEED = None
DEPOT_NODE = 2   # node_id van het laadstation/depot; pas aan naar jouw layout
GSE_SPEED = 4.0  # rijsnelheid van alle GSEs; batterijverbruik schaalt automatisch mee

# Visualisatie
plot_graph         = False
visualization      = True
real_minutes_per_pseudo_hour = 1.0  # 12 pseudo-hours -> 12 real minutes
gse_visual_max_step_distance = 0.2  # lagere waarde = vloeiendere GSE-beweging op het scherm
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


def assign_service_tasks(active_planes, auctioned_tasks, auction_system, heuristics, nodes_dict, t):
    unassigned_planes = [
        plane
        for plane in active_planes
        if plane.needs_service_assignment()
        and (plane.id, plane.next_service_type) not in auctioned_tasks
    ]

    if not unassigned_planes:
        return

    assignments = auction_system.allocate_tasks(unassigned_planes, heuristics)
    for gse, plane in assignments:
        service_type = plane.next_service_type
        if service_type is None:
            continue
        auctioned_tasks.add((plane.id, service_type))
        plane.mark_service_assigned(service_type, gse.id)
        gse.plan_service_task(plane, nodes_dict, heuristics, t, service_type=service_type)


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


def render_simulation_frame(map_properties, gse_lst, active_planes, t):
    """
    Teken de huidige simulatiestatus op het scherm.
    """
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
    return map_running(map_properties, current_states, gate_states, t)


def pace_simulation(sim_minutes, real_seconds_per_pseudo_minute, simulation_start_wall_time):
    """
    Houd de simulatie gelijk aan de gewenste wall-clock snelheid.
    """
    target_elapsed_seconds = sim_minutes * real_seconds_per_pseudo_minute
    remaining_sleep = target_elapsed_seconds - (timer.perf_counter() - simulation_start_wall_time)
    if remaining_sleep > 0:
        timer.sleep(remaining_sleep)

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
    unloading_duration_minutes,
    loading_duration_minutes,
    departure_delay_minutes,
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
              nodes_dict=nodes_dict, depot_node=DEPOT_NODE, speed=GSE_SPEED)
    gse_lst.append(gse)
print(f"[Init] {len(gse_lst)} GSEs aangemaakt met spawn nodes: {gse_spawn_config}")

# --- [NIEUW] Fleet Manager en Auction System ---
fleet_manager  = Fleet_manager(nodes_dict)
auction_system = AuctionSystem(gse_lst)

# Bijhouden welke servicetaken al een GSE toegewezen hebben gekregen
# zodat we niet bij elke tijdstap opnieuw een veiling houden voor dezelfde service.
auctioned_tasks = set()

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
real_seconds_per_pseudo_minute = real_minutes_per_pseudo_hour
simulation_start_wall_time = timer.perf_counter()

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

    plane_by_id = {plane.id: plane for plane in active_planes}

    # Rond services af waarvan de werktijd is verstreken.
    for gse in gse_lst:
        if gse.status == "working" and gse.work_end_time is not None and t >= gse.work_end_time - 1e-9:
            plane = plane_by_id.get(gse.assigned_plane_id)
            if plane is None:
                raise ValueError(
                    f"Assigned plane {gse.assigned_plane_id} for GSE {gse.id} is not active."
                )
            if gse.assigned_service_type == "load":
                plane.complete_current_service(t)
                print(
                    f"[Plane {plane.id}] t={t}: loading completed, departure possible at t={plane.departure_ready_time}"
                )
                gse.finish_working(nodes_dict, heuristics, t)
            elif gse.assigned_service_type == "unload":
                plane.complete_current_service(t)
                plane.mark_unloading_departed(gse.id)
                print(
                    f"[Plane {plane.id}] t={t}: unloading completed at gate, "
                    f"cargo en route to node {plane.cargo_to}; loading unlocks once GSE {gse.id} departs"
                )
                gse.plan_to_node(
                    plane.cargo_to,
                    nodes_dict,
                    heuristics,
                    t,
                    stage="unload_to_cargo",
                    label=f"cargo node {plane.cargo_to} with cargo from plane {plane.id}",
                )
            else:
                raise ValueError(
                    f"GSE {gse.id} has unknown service type '{gse.assigned_service_type}'."
                )

    # Vliegtuig vertrekt pas na completed loading + 5 minuten.
    despawned_ids = {
        plane.id
        for plane in active_planes
        if plane.ready_to_despawn(t)
    }
    if despawned_ids:
        for plane in active_planes:
            if plane.id in despawned_ids:
                plane.mark_departed(t)
                auctioned_tasks.discard((plane.id, "unload"))
                auctioned_tasks.discard((plane.id, "load"))
        active_planes = [plane for plane in active_planes if plane.id not in despawned_ids]
        plane_by_id = {plane.id: plane for plane in active_planes}

    # -------------------------------------------------------------------------
    # [NIEUW] Fleet Manager: update gate-bezettingskaart
    # -------------------------------------------------------------------------
    fleet_manager.update_gate_status(active_planes, aircraft_lst=gse_lst, t=t)

    # -------------------------------------------------------------------------
    # [NIEUW] Auction: wijs GSEs toe aan unload/load taken
    # -------------------------------------------------------------------------
    assign_service_tasks(
        active_planes,
        auctioned_tasks,
        auction_system,
        heuristics,
        nodes_dict,
        t,
    )

    # -------------------------------------------------------------------------
    # [NIEUW] GSEs die 'needs_charging' zijn sturen naar het depot
    # -------------------------------------------------------------------------
    for gse in gse_lst:
        if gse.status == "needs_charging":
            gse.go_charge(nodes_dict, heuristics, t)

    should_render_step = visualization and step_count % render_every_n_steps == 0
    if should_render_step:
        escape_pressed = render_simulation_frame(map_properties, gse_lst, active_planes, t)

    movement_substeps = 1
    if should_render_step:
        taxiing_gses = [gse for gse in gse_lst if gse.status == "taxiing"]
        if taxiing_gses:
            max_step_distance = max(gse.speed * dt for gse in taxiing_gses)
            movement_substeps = max(1, math.ceil(max_step_distance / gse_visual_max_step_distance))

    sub_dt = dt / movement_substeps
    for substep_index in range(movement_substeps):
        substep_time = round(t + (substep_index + 1) * sub_dt, 2)

        for gse in gse_lst:
            if gse.status == "taxiing":
                gse.move(sub_dt, substep_time)
            gse.update_soc(sub_dt)

        if should_render_step:
            escape_pressed = render_simulation_frame(map_properties, gse_lst, active_planes, substep_time)
            if escape_pressed:
                break
            pace_simulation(
                substep_time,
                real_seconds_per_pseudo_minute,
                simulation_start_wall_time,
            )

    if not should_render_step:
        pace_simulation(
            t + dt,
            real_seconds_per_pseudo_minute,
            simulation_start_wall_time,
        )

    # -------------------------------------------------------------------------
    # [NIEUW] GSEs die bij het depot zijn aangekomen: zet op 'charging'
    # (move() zet status al op 'charging' via _on_goal_reached, dit is een
    #  extra vangnet voor het geval de logica afwijkt)
    # -------------------------------------------------------------------------
    for gse in gse_lst:
        if gse.status == "arrived" and gse.current_node == DEPOT_NODE:
            gse.status = "charging"

    # Zodra de unload-GSE het plane node heeft verlaten, mag loading worden toegewezen.
    gse_by_id = {gse.id: gse for gse in gse_lst}
    for plane in active_planes:
        if plane.status != "awaiting_load_release" or plane.load_release_gse_id is None:
            continue
        unloading_gse = gse_by_id.get(plane.load_release_gse_id)
        if unloading_gse is None:
            raise ValueError(
                f"Unload-release GSE {plane.load_release_gse_id} for plane {plane.id} not found."
            )
        if unloading_gse.position != plane.xy_pos:
            plane.release_for_loading()
            print(
                f"[Plane {plane.id}] t={t}: unloading GSE left gate, loading can now be assigned"
            )

    assign_service_tasks(
        active_planes,
        auctioned_tasks,
        auction_system,
        heuristics,
        nodes_dict,
        t,
    )

    # Start nieuw gearriveerde unload/load services.
    plane_by_id = {plane.id: plane for plane in active_planes}
    for gse in gse_lst:
        if gse.status == "at_cargo_pickup":
            plane = plane_by_id.get(gse.assigned_plane_id)
            if plane is None:
                raise ValueError(
                    f"Assigned plane {gse.assigned_plane_id} for GSE {gse.id} is not active."
                )
            gse.plan_to_node(
                plane.node_id,
                nodes_dict,
                heuristics,
                t,
                stage="load_to_plane",
                label=f"plane {plane.id} with cargo from node {plane.cargo_from}",
            )
            print(
                f"[Plane {plane.id}] t={t}: cargo picked up at node {plane.cargo_from}, "
                f"en route to gate {plane.node_id}"
            )
        elif gse.status == "at_cargo_dropoff":
            plane = plane_by_id.get(gse.assigned_plane_id)
            if plane is None:
                raise ValueError(
                    f"Assigned plane {gse.assigned_plane_id} for GSE {gse.id} is not active."
                )
            print(
                f"[Plane {plane.id}] t={t}: unloaded cargo delivered to node {plane.cargo_to}"
            )
            gse.finish_working(nodes_dict, heuristics, t)
        elif gse.status == "working" and gse.work_end_time is None:
            plane = plane_by_id.get(gse.assigned_plane_id)
            if plane is None:
                raise ValueError(
                    f"Assigned plane {gse.assigned_plane_id} for GSE {gse.id} is not active."
                )
            gse.work_end_time = plane.start_service(gse.assigned_service_type, gse.id, t)
            print(
                f"[Plane {plane.id}] t={t}: {gse.assigned_service_type} started by GSE {gse.id}, "
                f"completes at t={gse.work_end_time}"
            )

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

print("\n--- Turnaround Times Planes ---")
total_turnaround_time = 0.0
completed_turnaround_count = 0
for plane in plane_schedule:
    turnaround_time = plane.turnaround_time
    if turnaround_time is None:
        print(f"Plane {plane.id}: turnaround incomplete (status={plane.status})")
        continue
    total_turnaround_time += turnaround_time
    completed_turnaround_count += 1
    print(f"Plane {plane.id}: turnaround time = {turnaround_time:.1f} minutes")

print(
    f"Totaal turnaround time ({completed_turnaround_count} voltooide planes): "
    f"{total_turnaround_time:.1f} minutes"
)
