"""
Run-me.py is the main file of the simulation. Run this file to run the simulation.
"""

from pathlib import Path
from datetime import datetime
import math
import random
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time as timer

from single_agent_planner import calc_heuristics
from GSE import GSE
from Plane import load_plane_schedule, build_plane_schedule_lookup, spawn_planes
from Fleet_manager import Fleet_manager
from auction_system import AuctionSystem
from cbs import resolve_conflicts  # <-- IMPORTED CBS MODULE

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOG_DIR  = PROJECT_ROOT / "run logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SIMULATION PARAMETERS  (pas hier aan)
# =============================================================================
nodes_file = "Data/nodes_EHAM.xlsx"
edges_file = "Data/edges_EHAM.xlsx"
plane_data_file = "Data/airport_schedule_24h.xlsx"

simulation_duration_hours = 12

# Service- en vertrektijden van vliegtuigen, in pseudo-minuten
unloading_duration_minutes = 15
loading_duration_minutes = 15
departure_delay_minutes = 5

# --- [NIEUW] GSE configuratie ---
GSE_COUNT = 8
GSE_RANDOM_SEED = None
GSE_SPEED = 4.0    # rijsnelheid van alle GSEs; batterijverbruik schaalt automatisch mee
GSE_ELECTRIC = False  # True = elektrisch (charge_duration=15 min, consumption=0.5%/unit)
               #       False = verbrandingsmotor (charge_duration=2 min, consumption=0.25%/unit)

# Auction wegingen (genormaliseerde bid-componenten)
AUCTION_ALPHA = 0.7
AUCTION_BETA = 0.3

# Visualisatie
plot_graph         = False
visualization      = True
real_minutes_per_pseudo_hour = 0.1  # 12 pseudo-hours -> 12 real minutes
gse_visual_max_step_distance = 0.2  # lagere waarde = vloeiendere GSE-beweging op het scherm
render_every_n_steps = 1

# GSE status tabel: print elke N simulatiestappen naar de terminal
status_print_interval = 10  # zet op 1 voor elke stap, hoger voor minder output


# =============================================================================
# FUNCTIE-DEFINITIES  
# =============================================================================

def import_layout(nodes_file, edges_file):
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

    return nodes_dict, edges_dict


def create_graph(nodes_dict, edges_dict, plot_graph=True):
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
    from visualization import map_running

    current_states = {}
    for gse in gse_lst:
        current_states[gse.id] = {
            "ac_id":   f"GSE {gse.id}",
            "xy_pos":  gse.position,
            "heading": gse.heading,
            "waiting": gse.waiting
        }

    gate_states = {
        plane.id: plane.to_gate_state()
        for plane in active_planes
    }
    return map_running(map_properties, current_states, gate_states, t)


def pace_simulation(sim_minutes, real_seconds_per_pseudo_minute, simulation_start_wall_time):
    target_elapsed_seconds = sim_minutes * real_seconds_per_pseudo_minute
    remaining_sleep = target_elapsed_seconds - (timer.perf_counter() - simulation_start_wall_time)
    if remaining_sleep > 0:
        timer.sleep(remaining_sleep)


def write_run_log(log_dir, gse_lst, plane_schedule, simulation_duration_hours, gse_speed,
                  gse_type_label, filepath=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_time = datetime.now()
    filename = filepath if filepath is not None \
        else log_dir / f"run_{run_time.strftime('%Y-%m-%d_%H-%M-%S')}_{gse_type_label}.log"

    completed = [(p.id, p.turnaround_time) for p in plane_schedule if p.turnaround_time is not None]
    total_turnaround = sum(tt for _, tt in completed)

    max_energy = max((g.total_energy_consumed for g in gse_lst), default=1.0) or 1.0

    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION RUN LOG")
    lines.append(f"GSE type   : {gse_type_label.upper()}")
    lines.append(f"Date/time  : {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Duration   : {simulation_duration_hours} hours ({simulation_duration_hours * 60} min)")
    lines.append(f"GSE count  : {len(gse_lst)}  |  speed: {gse_speed}")
    first_gse = gse_lst[0]
    lines.append(f"GSE params : charge_duration={first_gse.charge_duration:.0f} min  |  "
                 f"base_consumption={first_gse.base_consumption_rate}%/unit")
    lines.append("=" * 60)

    col = {"flight": 10, "tt": 12, "wu": 12, "wl": 11}
    hdr = (f"  {'Flight':<{col['flight']}}  {'Turnaround':>{col['tt']}}"
           f"  {'Wait unload':>{col['wu']}}  {'Wait load':>{col['wl']}}")
    sep = "  " + "-" * (sum(col.values()) + 8)

    lines.append("")
    lines.append("FLIGHT TIMES")
    lines.append(hdr)
    lines.append(sep)
    for plane in sorted(plane_schedule, key=lambda p: p.id):
        tt_str = f"{plane.turnaround_time:>10.1f} min" if plane.turnaround_time is not None else f"{'incomplete':>12}"
        wu_str = (f"{(plane.unload_start_time - plane.spawn_time):>10.1f} min"
                  if plane.unload_start_time is not None else f"{'incomplete':>12}")
        wl_str = (f"{(plane.load_start_time - plane.unloading_complete_time):>9.1f} min"
                  if plane.load_start_time is not None and plane.unloading_complete_time is not None
                  else f"{'incomplete':>11}")
        lines.append(f"  {plane.id:<{col['flight']}}  {tt_str}  {wu_str}  {wl_str}")
    lines.append(sep)
    lines.append(f"  {'Total (' + str(len(completed)) + ' completed)':<{col['flight']}}  "
                 f"{total_turnaround:>10.1f} min")
    if completed:
        lines.append(f"  {'Average':<{col['flight']}}  "
                     f"{total_turnaround / len(completed):>10.1f} min")

    lines.append("")
    lines.append("GSE UTILIZATION  (cumulative battery % consumed  |  tasks completed)")
    lines.append("-" * 60)
    for gse in sorted(gse_lst, key=lambda g: g.id):
        bar_width = 25
        filled = round(bar_width * gse.total_energy_consumed / max_energy)
        bar = "#" * filled + "-" * (bar_width - filled)
        lines.append(f"  GSE {gse.id:<3}  {gse.total_energy_consumed:>8.1f}%  [{bar}]  tasks: {gse.completed_tasks}")
    lines.append("=" * 60)

    mode = "a" if filepath is not None else "w"
    with open(filename, mode) as f:
        if filepath is not None:
            f.write("\n")
        f.write("\n".join(lines) + "\n")

    print(f"[Log] Run log saved to: {filename}")
    return filename


def _gse_task_description(gse):
    if gse.status == "charging":
        return f"charging ({gse.charge_time_elapsed:.0f}/{gse.charge_duration:.0f} min)"
    if gse.status == "needs_charging":
        return "waiting to charge"
    if gse.task_stage == "to_depot":
        return "heading to depot"
    if gse.assigned_plane_id is not None:
        p = f"plane {gse.assigned_plane_id}"
        stage_labels = {
            "load_to_cargo":    f"load: fetching cargo  ({p})",
            "load_to_plane":    f"load: delivering to {p}",
            "unload_to_plane":  f"unload: going to {p}",
            "unload_to_cargo":  f"unload: delivering cargo ({p})",
            "service_to_plane": f"service: going to {p}",
        }
        if gse.task_stage in stage_labels:
            return stage_labels[gse.task_stage]
        if gse.status == "working":
            return f"{gse.assigned_service_type} on {p}"
        if gse.status in ("at_cargo_pickup", "at_cargo_dropoff"):
            return f"{gse.status.replace('_', ' ')} ({p})"
    return "-"


def print_gse_status_table(gse_lst, t, prev_line_count=0):
    col_task = 36
    header = (f"  {'GSE':>3} │ {'SoC':>6} │ {'Available':>9} │ {'Status':<20} │ Task")
    sep    = "  " + "─" * (3 + 3 + 8 + 3 + 11 + 3 + 22 + 3 + col_task)

    lines = [
        f"  ── GSE Status  t = {t:.0f} min {'─' * 40}",
        header,
        sep,
    ]
    for gse in gse_lst:
        avail = "yes" if gse.status == "available" else "no"
        task  = _gse_task_description(gse)
        lines.append(
            f"  {gse.id:>3} │ {gse.soc:>5.1f}% │ {avail:>9} │ {gse.status:<20} │ {task}"
        )
    lines.append(sep)

    if prev_line_count > 0:
        sys.stdout.write(f"\033[{prev_line_count}A\033[J")

    output = "\n".join(lines) + "\n"
    sys.stdout.write(output)
    sys.stdout.flush()
    return len(lines)

# =============================================================================
# INITIALISATIE 
# =============================================================================
nodes_dict, edges_dict = import_layout(nodes_file, edges_file)
graph      = create_graph(nodes_dict, edges_dict, plot_graph)
heuristics = calc_heuristics(graph, nodes_dict)
MAX_SHORTEST_PATH_DISTANCE = max(
    (distance for distances in heuristics.values() for distance in distances.values()),
    default=1.0,
)
if MAX_SHORTEST_PATH_DISTANCE <= 0:
    MAX_SHORTEST_PATH_DISTANCE = 1.0

charging_node_ids = [nid for nid, props in nodes_dict.items() if props["type"] == "charging"]
if not charging_node_ids:
    raise ValueError("Geen oplaadstation nodes gevonden in de layout (type='charging').")
print(f"[Init] Oplaadstations gevonden: nodes {charging_node_ids}")


# =============================================================================
# SIMULATIEFUNCTIE
# =============================================================================
def run_simulation(charge_duration, base_consumption_rate, gse_type_label, log_filepath=None,
                   gse_spawn_config=None, *, enable_visualization=None, pace=True,
                   show_status=True):
    print(f"\n{'=' * 60}")
    print(f"  Starting {gse_type_label.upper()} simulation")
    print(f"  charge_duration={charge_duration} min  |  base_consumption={base_consumption_rate}%/unit")
    print(f"{'=' * 60}")

    visualize = visualization if enable_visualization is None else bool(enable_visualization)

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

    if gse_spawn_config is None:
        gse_spawn_config = build_random_gse_spawn_config(nodes_dict, GSE_COUNT, seed=GSE_RANDOM_SEED)
    gse_lst = []
    for gse_id, start_node in gse_spawn_config:
        gse = GSE(gse_id=gse_id, start_node=start_node,
                  nodes_dict=nodes_dict, charging_nodes=charging_node_ids, speed=GSE_SPEED)
        gse.charge_duration       = charge_duration
        gse.base_consumption_rate = base_consumption_rate
        gse.set_speed(GSE_SPEED)
        gse_lst.append(gse)
    print(f"[Init] {len(gse_lst)} {gse_type_label} GSEs aangemaakt met spawn nodes: {gse_spawn_config}")

    fleet_manager  = Fleet_manager(nodes_dict)
    auction_system = AuctionSystem(
        gse_lst,
        alpha=AUCTION_ALPHA,
        beta=AUCTION_BETA,
        max_shortest_path_distance=MAX_SHORTEST_PATH_DISTANCE,
    )
    print(
        "[Init] Auction weights: "
        f"alpha={AUCTION_ALPHA:.3f}, beta={AUCTION_BETA:.3f}, "
        f"d_max={MAX_SHORTEST_PATH_DISTANCE:.3f}"
    )
    auctioned_tasks = set()

    if visualize:
        from visualization import map_initialization

        map_properties = map_initialization(nodes_dict, edges_dict)

    # =========================================================================
    # SIMULATIELUS
    # =========================================================================
    running                        = True
    escape_pressed                 = False
    time_end                       = simulation_duration_hours * 60
    dt                             = 1.0
    t                              = 0
    step_count                     = 0
    real_seconds_per_pseudo_minute = real_minutes_per_pseudo_hour
    simulation_start_wall_time     = timer.perf_counter()
    _status_table_lines            = 0

    print("Simulation Started")
    while running:
        t = round(t, 2)

        if t >= time_end or escape_pressed:
            running = False
            if visualize:
                import pygame as pg
                pg.quit()
            print("Simulation Stopped")
            break

        new_planes = spawn_planes(t, plane_schedule_lookup)
        if new_planes:
            active_planes.extend(new_planes)

        plane_by_id = {plane.id: plane for plane in active_planes}

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

        despawned_ids = {plane.id for plane in active_planes if plane.ready_to_despawn(t)}
        if despawned_ids:
            for plane in active_planes:
                if plane.id in despawned_ids:
                    plane.mark_departed(t)
                    auctioned_tasks.discard((plane.id, "unload"))
                    auctioned_tasks.discard((plane.id, "load"))
            active_planes = [plane for plane in active_planes if plane.id not in despawned_ids]
            plane_by_id   = {plane.id: plane for plane in active_planes}

        fleet_manager.update_gate_status(active_planes, aircraft_lst=gse_lst, t=t)

        assign_service_tasks(active_planes, auctioned_tasks, auction_system, heuristics, nodes_dict, t)

        charging_load = {}
        for gse in gse_lst:
            if gse.status == "charging" or (gse.status == "taxiing" and gse.task_stage == "to_depot"):
                if gse.goal is not None:
                    charging_load[gse.goal] = charging_load.get(gse.goal, 0) + 1

        for gse in gse_lst:
            if gse.status == "needs_charging":
                if gse.assigned_plane_id is not None:
                    plane = plane_by_id.get(gse.assigned_plane_id)
                    if (plane is not None
                            and plane.status in ("unload_assigned", "load_assigned")
                            and plane.assigned_gse_id == gse.id):
                        service_type = "unload" if plane.status == "unload_assigned" else "load"
                        plane.status = "awaiting_unload" if service_type == "unload" else "awaiting_load"
                        plane.assigned_gse_id = None
                        auctioned_tasks.discard((plane.id, service_type))
                        print(f"[GSE {gse.id}] t={t}: abandoned {service_type} task for plane {plane.id} (low battery), re-queuing for auction")
                    gse.assigned_plane_id = None
                gse.go_charge(nodes_dict, heuristics, t, busy_charging_nodes=charging_load)
                if gse.goal is not None:
                    charging_load[gse.goal] = charging_load.get(gse.goal, 0) + 1

        should_render_step = visualize and step_count % render_every_n_steps == 0
        if should_render_step:
            escape_pressed = render_simulation_frame(map_properties, gse_lst, active_planes, t)
        
        # --- CBS: Conflict Resolution ---
        resolve_conflicts(gse_lst, nodes_dict, heuristics, t)

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
                if gse.status == "taxiing" and not gse.waiting:
                    gse.move(sub_dt, substep_time)
                gse.update_soc(sub_dt)

            if should_render_step:
                escape_pressed = render_simulation_frame(map_properties, gse_lst, active_planes, substep_time)
                if escape_pressed:
                    break
                if pace:
                    pace_simulation(substep_time, real_seconds_per_pseudo_minute, simulation_start_wall_time)

        if not should_render_step and pace:
            pace_simulation(t + dt, real_seconds_per_pseudo_minute, simulation_start_wall_time)

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
                print(f"[Plane {plane.id}] t={t}: unloading GSE left gate, loading can now be assigned")

        assign_service_tasks(active_planes, auctioned_tasks, auction_system, heuristics, nodes_dict, t)

        plane_by_id = {plane.id: plane for plane in active_planes}
        for gse in gse_lst:
            if gse.status == "at_cargo_pickup":
                plane = plane_by_id.get(gse.assigned_plane_id)
                if plane is None:
                    print(f"[GSE {gse.id}] t={t}: assigned plane {gse.assigned_plane_id} no longer active, releasing GSE")
                    gse.finish_working(nodes_dict, heuristics, t)
                    continue
                gse.plan_to_node(
                    plane.node_id, nodes_dict, heuristics, t,
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
                    print(f"[GSE {gse.id}] t={t}: assigned plane {gse.assigned_plane_id} no longer active, releasing GSE")
                    gse.finish_working(nodes_dict, heuristics, t)
                    continue
                print(f"[Plane {plane.id}] t={t}: unloaded cargo delivered to node {plane.cargo_to}")
                gse.finish_working(nodes_dict, heuristics, t)
            elif gse.status == "working" and gse.work_end_time is None:
                plane = plane_by_id.get(gse.assigned_plane_id)
                if plane is None:
                    print(f"[GSE {gse.id}] t={t}: assigned plane {gse.assigned_plane_id} no longer active, releasing GSE")
                    gse.finish_working(nodes_dict, heuristics, t)
                    continue
                gse.work_end_time = plane.start_service(gse.assigned_service_type, gse.id, t)
                print(
                    f"[Plane {plane.id}] t={t}: {gse.assigned_service_type} started by GSE {gse.id}, "
                    f"completes at t={gse.work_end_time}"
                )

        if show_status and step_count % status_print_interval == 0:
            _status_table_lines = print_gse_status_table(gse_lst, t, _status_table_lines)

        t = t + dt
        step_count += 1

    write_run_log(LOG_DIR, gse_lst, plane_schedule, simulation_duration_hours, GSE_SPEED,
                  gse_type_label, filepath=log_filepath)


# =============================================================================
# UITVOERING
# =============================================================================
if __name__ == "__main__":
    if GSE_ELECTRIC:
        run_simulation(15.0, 0.5, "electric")
    else:
        run_simulation(2.0, 0.25, "gas")
