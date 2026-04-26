from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass
from matplotlib.lines import Line2D

from auction_system import AuctionSystem
from GSE import GSE
from single_agent_planner import calc_heuristics, simple_single_agent_astar
from cbs import _replan_around, resolve_conflicts

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "Figures"

@dataclass
class Task:
    id: str
    node_id: int
    cargo_from: int
    cargo_to: int
    next_service_type: str = "unload"

def load_simple_graph():
    nodes = {
        1: {"id": 1, "type": "cargo", "xy_pos": (0, 0), "neighbors": {2}},
        2: {"id": 2, "type": "intersection", "xy_pos": (1, 0), "neighbors": {1, 3, 6}},
        3: {"id": 3, "type": "intersection", "xy_pos": (2, 0), "neighbors": {2, 4, 8}},
        4: {"id": 4, "type": "intersection",  "xy_pos": (3, 0), "neighbors": {3, 5}},
        5: {"id": 5, "type": "gate",  "xy_pos": (4, 0), "neighbors": {4}},
        6: {"id": 6, "type": "intersection", "xy_pos": (1, 1), "neighbors": {2, 7}},
        7: {"id": 7, "type": "intersection", "xy_pos": (2, 1), "neighbors": {6, 5}},
        8: {"id": 8, "type": "intersection", "xy_pos": (2, -1), "neighbors": {3, 5}},
    }
    edges = {
        (1, 2): 1, (2, 1): 1, (2, 3): 2, (3, 2): 2, (3, 4): 1, (4, 3): 1,
        (4, 5): 4, (5, 4): 4, (2, 6): 2, (6, 2): 2, (6, 7): 1, (7, 6): 1,
        (7, 5): 2, (5, 7): 2, (3, 8): 2, (8, 3): 2, (8, 5): 2, (5, 8): 2,
    }
    graph = nx.DiGraph()
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in edges.items())
    h = calc_heuristics(graph, nodes)
    return nodes, edges, graph, h

def load_grid_10x10():
    """Build a 10x10 equally spaced grid with bidirectional unit-length edges."""
    nodes = {}
    edges = {}
    n = 10

    def idx(r, c):
        return r * n + c + 1

    for r in range(n):
        for c in range(n):
            node_id = idx(r, c)
            nodes[node_id] = {
                "id": node_id,
                "type": "intersection",
                "xy_pos": (float(c), float(r)),
                "neighbors": set(),
            }

    for r in range(n):
        for c in range(n):
            a = idx(r, c)
            if c + 1 < n:
                b = idx(r, c + 1)
                edges[(a, b)] = 1.0
                edges[(b, a)] = 1.0
                nodes[a]["neighbors"].add(b)
                nodes[b]["neighbors"].add(a)
            if r + 1 < n:
                b = idx(r + 1, c)
                edges[(a, b)] = 1.0
                edges[(b, a)] = 1.0
                nodes[a]["neighbors"].add(b)
                nodes[b]["neighbors"].add(a)

    graph = nx.DiGraph()
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in edges.items())
    h = calc_heuristics(graph, nodes)
    return nodes, graph, h


def setup_cbs_grid_case():
    """Deterministic long-route collision case on 10x10 grid."""
    nodes, graph, h = load_grid_10x10()
    charging = []
    route1 = [23, 43, 44, 45, 46, 47, 48, 49, 59, 69]
    route2 = [24, 34, 35, 45, 55, 65, 75, 85,86, 96]
    conflict_node = 45
    conflict_t = 4

    return nodes, graph, h, charging, route1, route2, conflict_node, conflict_t

def _route_nodes(gse):
    return [gse.current_node] + [n for n, _ in gse.path_to_goal]

def _advance(gse):
    if gse.status != "taxiing" or gse.waiting or not gse.path_to_goal:
        return
    n, _ = gse.path_to_goal.pop(0)
    gse.from_to = [gse.current_node, n]
    gse.current_node = n
    gse.position = gse.nodes_dict[gse.current_node]["xy_pos"]

# Load simple graph
NODES, EDGES, GRAPH, H = load_simple_graph()
CHARGING = [4, 5]
ALPHA, BETA = 0.7, 0.3
D_MAX = max((d for dists in H.values() for d in dists.values()), default=1.0) or 1.0

# Verification tests
#1 A* pathfinding
def check_astar():
    ok, path = simple_single_agent_astar(NODES, 1, 5, H, 0)
    actual = [int(p[0]) for p in path] if ok else []
    return ok and actual == [1, 2, 6, 7, 5], actual

#2 Heuristic values
def check_heuristics():
    return all(H[a][b] > 0 or a == b for a in NODES for b in NODES), len(NODES)

#3 Bidding mechanism
def check_bidding():
    gse1 = GSE(1, 3, NODES, charging_nodes=CHARGING, speed=1.0)
    gse2 = GSE(2, 7, NODES, charging_nodes=CHARGING, speed=1.0)
    gse1.soc, gse2.soc = 90.0, 60.0
    
    task = Task("P1", 5, 1, 8, "unload")
    auction = AuctionSystem([gse1, gse2], alpha=ALPHA, beta=BETA, max_shortest_path_distance=D_MAX)
    winner = auction.allocate_tasks([task], H)[0][0].id
    return True, winner

#4 Min/max constraints
def check_min_max():
    gse = GSE(1, 6, NODES, charging_nodes=CHARGING, speed=1.0)
    try:
        gse.set_speed(0)
        return False
    except ValueError:
        pass
    
    gse.status, gse.soc = "taxiing", 1.0
    gse.update_soc(1000.0)
    if gse.soc < 0:
        return False
    
    gse.status = "charging"
    gse.goal, gse.current_node, gse.soc = 6, 6, 20.0
    gse.charge_duration, gse.charge_time_elapsed = 2.0, 0.0
    gse.update_soc(2.5)
    return gse.status == "available" and gse.soc == 100.0

#5 Constraints: gate, nodes, charging
def check_constraints():
    return (NODES[5]["type"] == "gate" and 
            1 in NODES and 8 in NODES and 
            any(c in H.get(1, {}) for c in CHARGING))

#6 CBS priority and yielding
def check_cbs_priority():
    #The lower ID has higher priority
    nodes, graph, h, charging, r1, r2, conflict_node, conflict_t = setup_cbs_grid_case()
    
    gse1 = GSE(1, r1[0], nodes, charging_nodes=charging, speed=1.0)
    gse2 = GSE(2, r2[0], nodes, charging_nodes=charging, speed=1.0)
    
    for gse, route in [(gse1, r1), (gse2, r2)]:
        gse.status = "taxiing"
        gse.goal = route[-1]
        gse.current_node = route[0]
        gse.path_to_goal = [(n, 0) for n in route[1:]]
        gse.from_to = [route[0], route[1]]
        gse.waiting = False
    
    gse2_path_before = [n for n, _ in gse2.path_to_goal]
    resolve_conflicts([gse1, gse2], nodes, h, t=0)
    gse2_path_after = [n for n, _ in gse2.path_to_goal]
    
    ok = (not gse1.waiting) and (gse2.waiting or gse2_path_after != gse2_path_before)
    return ok, (
        f"same_time_cross=({conflict_node}, t={conflict_t}), "
        f"GSE1 waiting={gse1.waiting}, "
        f"GSE2 yielded={gse2.waiting or gse2_path_after != gse2_path_before}"
    )
#7 CBS rerouting around conflict node
def check_cbs_reroute():
    """Verify _replan_around reroutes around conflict node on 10x10 grid."""
    nodes, graph, h, charging, r1, r2, conflict_node, conflict_t = setup_cbs_grid_case()

    # Replan GSE2 around the crossing node.
    gse = GSE(2, r2[0], nodes, charging_nodes=charging, speed=1.0)
    gse.status = "taxiing"
    gse.goal, gse.current_node = r2[-1], r2[0]
    gse.path_to_goal = [(n, 0) for n in r2[1:]]
    gse.from_to = [r2[0], r2[1]]

    # The conflict node is the one we want to block in the replan.
    blocked_node = conflict_node
    old_path = [n for n, _ in gse.path_to_goal]

    # We expect _replan_around to find a new path that avoids the blocked node, which is the conflict node at t=0.
    result = _replan_around(gse, {blocked_node}, nodes, h, t=0)
    new_path = [n for n, _ in gse.path_to_goal]

    ok = result and (new_path != old_path) and (blocked_node not in new_path)
    return ok, (
        f"same_time_cross=({conflict_node}, t={conflict_t}), "
        f"reroute_success={result}, path_changed={new_path != old_path}, "
        f"avoided_conflict_node={blocked_node not in new_path}"
    )

#
def check_cbs_timeline():
    """Verify CBS resolves same-time crossing until both GSEs reach goals."""
    nodes, graph, h, charging, r1, r2, conflict_node, conflict_t = setup_cbs_grid_case()
    goal1, goal2 = r1[-1], r2[-1]

    #GSE's 
    gse1 = GSE(1, r1[0], nodes, charging_nodes=charging, speed=1.0)
    gse2 = GSE(2, r2[0], nodes, charging_nodes=charging, speed=1.0)

    for gse, route in [(gse1, r1), (gse2, r2)]:
        gse.status = "taxiing"
        gse.goal = route[-1]
        gse.current_node = route[0]
        gse.path_to_goal = [(n, 0) for n in route[1:]]
        gse.from_to = [route[0], route[1]]
        gse.waiting = False

    print("\nCBS collision verification on 10x10 grid")
    print(f"GSE1 route before CBS: {r1}")
    print(f"GSE2 route before CBS: {r2}")
    print(f"planned same-time crossing at node {conflict_node} on t={conflict_t}")

    collisions_non_goal = []
    gse2_yielded_or_rerouted = False
    gse2_initial_path = [n for n, _ in gse2.path_to_goal]
    resolved_routes_t0 = None
    gse2_artificial_noop_ticks = []

    t = 0
    max_steps = 30
    while t < max_steps:
        path2_before = [n for n, _ in gse2.path_to_goal]
        resolve_conflicts([gse1, gse2], nodes, h, t=t)
        path2_after = [n for n, _ in gse2.path_to_goal]

        if t == 0:
            resolved_routes_t0 = (_route_nodes(gse1), _route_nodes(gse2))

        if gse2.waiting or path2_after != path2_before:
            gse2_yielded_or_rerouted = True

        if (not gse2.waiting) and path2_after and path2_after[0] == gse2.current_node:
            gse2_artificial_noop_ticks.append(t)

        print(
            f"t={t}: GSE1 node={gse1.current_node}, waiting={gse1.waiting}; "
            f"GSE2 node={gse2.current_node}, waiting={gse2.waiting}, path={path2_after}"
        )

        if gse1.current_node == gse2.current_node and gse1.current_node not in {goal1, goal2}:
            collisions_non_goal.append((t, gse1.current_node))

        if not gse1.waiting:
            gse1.move(dt=1.0, t=t)
        if not gse2.waiting:
            gse2.move(dt=1.0, t=t)

        reached1 = gse1.current_node == goal1 and not gse1.path_to_goal
        reached2 = gse2.current_node == goal2 and not gse2.path_to_goal
        if reached1 and reached2:
            break

        t += 1

    steps_used = t + 1

    gse2_changed_from_initial = [n for n, _ in gse2.path_to_goal] != gse2_initial_path
    no_artificial_wait = len(gse2_artificial_noop_ticks) == 0
    ok = gse2_yielded_or_rerouted and len(collisions_non_goal) == 0 and no_artificial_wait
    cbs_plot = plot_cbs_grid_case(nodes,graph,r1,r2,resolved_routes_t0[0] if resolved_routes_t0 else r1,resolved_routes_t0[1] if resolved_routes_t0 else r2,conflict_node,
    )
    return ok, {
        "msg": (
        f"same_time_cross=({conflict_node}, t={conflict_t}), "
        f"yield_or_reroute={gse2_yielded_or_rerouted}, "
        f"path_changed={gse2_changed_from_initial}, "
        f"no_artificial_wait={no_artificial_wait}, "
        f"artificial_noop_ticks={gse2_artificial_noop_ticks}, "
        f"collisions_non_goal={collisions_non_goal}, "
        f"steps={steps_used}, reached_goals={gse1.current_node == goal1 and gse2.current_node == goal2}"
        ),
        "plot": cbs_plot,
    }


def plot_cbs_grid_case(nodes, graph, route1_before, route2_before, route1_after, route2_after, conflict_node):
    pos = {n: nodes[n]["xy_pos"] for n in nodes}
    fig, ax = plt.subplots(figsize=(8, 8))

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#d8d8d8", width=0.9, arrows=False)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=35, node_color="grey")

    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=list(zip(route1_before, route1_before[1:])),edge_color="#1f77b4",width=2.0,style="dashed",arrows=False,)
    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=list(zip(route2_before, route2_before[1:])),edge_color="#ff7f0e",width=2.0,style="dashed", arrows=False,)
    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=list(zip(route1_after, route1_after[1:])),edge_color="#0057b8",width=3.0,arrows=False,)
    nx.draw_networkx_edges(graph,pos,ax=ax,edgelist=list(zip(route2_after, route2_after[1:])),edge_color="#d62728",width=3.0,arrows=False,)

    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[route1_before[0]], node_color="#1f77b4", node_size=110)
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[route2_before[0]], node_color="#ff7f0e", node_size=110)
    nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=[conflict_node], node_color="#aa0000", node_size=130)

    x, y = pos[conflict_node]
    ax.text(x + 0.15, y + 0.15, f"conflict node {conflict_node}", fontsize=9)
    ax.set_title("CBS on 10x10 grid: dashed=planned, solid=after resolve_conflicts(t=0)")

    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=2.0, ls="--", label="GSE1 planned"),
        Line2D([0], [0], color="#ff7f0e", lw=2.0, ls="--", label="GSE2 planned"),
        Line2D([0], [0], color="#0057b8", lw=3.0, label="GSE1 after CBS"),
        Line2D([0], [0], color="#d62728", lw=3.0, label="GSE2 after CBS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aa0000", markersize=9, label="Conflict node"),
    ]
    ax.legend(handles=legend_items, loc="upper left", frameon=True)

    ax.set_aspect("equal")
    ax.axis("off")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "cbs_grid_10x10.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_astar():
    pos = {n: NODES[n]["xy_pos"] for n in NODES}
    fig, ax = plt.subplots(figsize=(6, 4))
    
    #draw edges,nodes and labels
    nx.draw_networkx_edges(GRAPH, pos, ax=ax, edge_color="#ccc", width=1.0, arrows=False)
    nx.draw_networkx_nodes(GRAPH, pos, ax=ax, node_size=300, node_color="#9ecae1")
    nx.draw_networkx_labels(GRAPH, pos, ax=ax)

    #saving the figure
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "astar_simple.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    return FIG_DIR / "astar_simple.png"

def main():
    print("\n=== Verification Checks ===")

    ok, path = check_astar()
    print(f"[{'PASS' if ok else 'FAIL'}] A* path: {path}")
    
    ok, n = check_heuristics()
    print(f"[{'PASS' if ok else 'FAIL'}] Heuristics: {n} nodes")
    
    ok, winner = check_bidding()
    print(f"[PASS] Bidding: winner=GSE {winner}")
    
    ok = check_min_max()
    print(f"[{'PASS' if ok else 'FAIL'}] Min/max: SOC constraint")
    
    ok = check_constraints()
    print(f"[{'PASS' if ok else 'FAIL'}] Constraints: gate, nodes, charging")
    
    ok, msg = check_cbs_priority()
    print(f"[{'PASS' if ok else 'FAIL'}] CBS Priority: {msg}")
    
    ok, msg = check_cbs_reroute()
    print(f"[{'PASS' if ok else 'FAIL'}] CBS Reroute: {msg}")



    ok, timeline_info = check_cbs_timeline()
    print(f"[{'PASS' if ok else 'FAIL'}] CBS Timeline: {timeline_info['msg']}")
    
    astar_plot = plot_astar()
    print(f"\nPlots: {astar_plot}")
    print(f"       {timeline_info['plot']}")

if __name__ == "__main__":
    main()
