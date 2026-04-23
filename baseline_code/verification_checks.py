from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from auction_system import AuctionSystem
from GSE import GSE
from single_agent_planner import calc_heuristics, simple_single_agent_astar


BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "Figures"
LOG_DIR = BASE / "Data" / "verification_logs"


@dataclass
class Task:
    id: str
    node_id: int
    cargo_from: int
    cargo_to: int
    next_service_type: str


def simple_dataset():
    nodes = {
        1: {"id": 1, "type": "cargo", "xy_pos": (0, 0), "neighbors": {2}},
        2: {"id": 2, "type": "cargo", "xy_pos": (1, 0), "neighbors": {1, 3, 6}},
        3: {"id": 3, "type": "cargo", "xy_pos": (2, 0), "neighbors": {2, 4, 8}},
        4: {"id": 4, "type": "gate",  "xy_pos": (3, 0), "neighbors": {3, 5}},
        5: {"id": 5, "type": "gate",  "xy_pos": (4, 0), "neighbors": {4}},
        6: {"id": 6, "type": "charging", "xy_pos": (1, 1), "neighbors": {2, 7}},
        7: {"id": 7, "type": "charging", "xy_pos": (2, 1), "neighbors": {6, 5}},
        8: {"id": 8, "type": "cargo", "xy_pos": (2, -1), "neighbors": {3, 5}},
    }
    edges = {
        (1, 2): 1, (2, 1): 1,
        (2, 3): 2, (3, 2): 2,
        (3, 4): 1, (4, 3): 1,
        (4, 5): 4, (5, 4): 4,
        (2, 6): 2, (6, 2): 2,
        (6, 7): 1, (7, 6): 1,
        (7, 5): 2, (5, 7): 2,
        (3, 8): 2, (8, 3): 2,
        (8, 5): 2, (5, 8): 2,
    }
    graph = nx.DiGraph()
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in edges.items())
    heuristics = calc_heuristics(graph, nodes)
    return nodes, edges, graph, heuristics


NODES, EDGES, GRAPH, H = simple_dataset()
CHARGING = [4, 5]


def path_weight(path):
    return sum(EDGES[(int(path[i][0]), int(path[i + 1][0]))] for i in range(len(path) - 1))


def check_astar():
    ok, path = simple_single_agent_astar(NODES, 1, 5, H, 0.0)
    expected = [1, 2, 6, 7, 5]
    actual = [int(p[0]) for p in path] if ok else []
    return ok and actual == expected, {
        "actual_path": actual,
        "expected_path": expected,
        "length": path_weight(path) if ok else None,
    }


def check_heuristics():
    return all(H[a][b] > 0 or a == b for a in NODES for b in NODES), {"nodes": len(NODES), "pairs": len(NODES) ** 2}


def check_bidding():
    # Scenario 1: position affects bids/winner (different start nodes, similar SoC)
    gse1 = GSE(1, 3, NODES, charging_nodes=CHARGING, speed=1.0)
    gse2 = GSE(2, 7, NODES, charging_nodes=CHARGING, speed=1.0)
    gse1.soc = 90.0
    gse2.soc = 60.0
    task = Task("P1", 5, 1, 8, "unload")
    auction = AuctionSystem([gse1, gse2])

    bids = []
    for gse in [gse1, gse2]:
        bid = gse.calculate_bid(task.node_id, H, second_node_id=task.cargo_to)
        bids.append({"task_id": task.id, "gse_id": gse.id, "bid": bid, "soc": gse.soc, "start_node": gse.current_node})

    winner = auction.allocate_tasks([task], H)[0][0].id
    position_ok = winner == min(bids, key=lambda x: x["bid"])["gse_id"]

    # Scenario 2: SoC affects bids (same start node, only SoC differs)
    gse_high = GSE(3, 6, NODES, charging_nodes=CHARGING, speed=1.0)
    gse_low = GSE(4, 6, NODES, charging_nodes=CHARGING, speed=1.0)
    gse_high.soc = 90.0
    gse_low.soc = 30.0
    bid_high = gse_high.calculate_bid(task.node_id, H, second_node_id=task.cargo_to)
    bid_low = gse_low.calculate_bid(task.node_id, H, second_node_id=task.cargo_to)
    soc_effect_ok = bid_high < bid_low

    # Scenario 3: very low SoC blocks bidding
    gse_block = GSE(5, 6, NODES, charging_nodes=CHARGING, speed=1.0)
    gse_block.soc = 5.0
    low_soc_bid = gse_block.calculate_bid(task.node_id, H, second_node_id=task.cargo_to)
    low_soc_ok = low_soc_bid == float("inf")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = position_ok and soc_effect_ok and low_soc_ok
    return all_ok, {
        "winner": winner,
        "bids": bids,
        "position_ok": position_ok,
        "soc_effect_ok": soc_effect_ok,
        "low_soc_ok": low_soc_ok,
        "bid_high_soc": bid_high,
        "bid_low_soc": bid_low,
        "low_soc_bid": low_soc_bid,
    }


def check_min_max():
    gse = GSE(1, 6, NODES, charging_nodes=CHARGING, speed=1.0)
    try:
        gse.set_speed(0)
        return False, {"reason": "set_speed(0) should fail"}
    except ValueError:
        pass

    gse.status = "taxiing"
    gse.soc = 1.0
    gse.update_soc(1000.0)
    if gse.soc < 0:
        return False, {"reason": f"soc went below zero: {gse.soc}"}

    gse.status = "charging"
    gse.goal = 6
    gse.current_node = 6
    gse.soc = 20.0
    gse.charge_duration = 2.0
    gse.charge_time_elapsed = 0.0
    gse.update_soc(2.5)
    return gse.status == "available" and gse.soc == 100.0, {"soc": gse.soc, "status": gse.status}


def check_constraints():
    task = Task("P1", 5, 1, 8, "unload")
    gate_ok = NODES[task.node_id]["type"] == "gate"
    cargo_ok = task.cargo_from in NODES and task.cargo_to in NODES
    charge_ok = any(c in H.get(task.cargo_from, {}) for c in CHARGING)
    return gate_ok and cargo_ok and charge_ok, {"gate_ok": gate_ok, "cargo_ok": cargo_ok, "charge_ok": charge_ok}


def plot_astar(actual_path, expected_path=None):
    pos = {n: NODES[n]["xy_pos"] for n in NODES}
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw_networkx_edges(GRAPH, pos, ax=ax, edge_color="#cccccc", width=1.0, arrows=False)
    nx.draw_networkx_nodes(GRAPH, pos, ax=ax, node_size=300, node_color="#9ecae1")
    nx.draw_networkx_labels(GRAPH, pos, ax=ax)
    nx.draw_networkx_edge_labels(GRAPH, pos, ax=ax, edge_labels={(a, b): w for (a, b), w in EDGES.items()}, font_size=7)

    if expected_path and len(expected_path) > 1:
        nx.draw_networkx_edges(
            GRAPH,
            pos,
            ax=ax,
            edgelist=list(zip(expected_path, expected_path[1:])),
            edge_color="#1f77b4",
            width=2.0,
            style="dashed",
            arrows=False,
        )

    if actual_path and len(actual_path) > 1:
        nx.draw_networkx_edges(
            GRAPH,
            pos,
            ax=ax,
            edgelist=list(zip(actual_path, actual_path[1:])),
            edge_color="#d62728",
            width=2.8,
            arrows=False,
        )

    ax.set_title(f"A* actual path: {actual_path}")
    ax.axis("off")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "astar_simple.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out

def plot_bidding_graph(gse1, gse2, task, winner, bidding_info):
    pos = {n: NODES[n]["xy_pos"] for n in NODES}
    fig, ax = plt.subplots(figsize=(7, 4))
    nx.draw_networkx_edges(GRAPH, pos, ax=ax, edge_color="#d0d0d0", width=1.0, arrows=False)
    nx.draw_networkx_edge_labels(GRAPH, pos, ax=ax, edge_labels={(a, b): w for (a, b), w in EDGES.items()}, font_size=7)
    nx.draw_networkx_nodes(GRAPH, pos, ax=ax, node_size=260, node_color="#9ecae1")
    nx.draw_networkx_labels(GRAPH, pos, ax=ax)

    bid_by_gse = {b["gse_id"]: b for b in bidding_info["bids"]}

    gse1_xy = NODES[gse1.current_node]["xy_pos"]
    gse2_xy = NODES[gse2.current_node]["xy_pos"]
    color_1 = "#ff7f0e" if gse1.id == winner else "#2ca02c"
    color_2 = "#ff7f0e" if gse2.id == winner else "#2ca02c"
    ax.scatter(*gse1_xy, s=260, c=color_1, marker="s", zorder=5)
    ax.scatter(*gse2_xy, s=260, c=color_2, marker="s", zorder=5)

    gse1_info = bid_by_gse.get(gse1.id, {})
    gse2_info = bid_by_gse.get(gse2.id, {})
    ax.text(
        gse1_xy[0] + 0.06,
        gse1_xy[1] + 0.06,
        f"GSE {gse1.id}\nbid={gse1_info.get('bid')}\nSoC={gse1_info.get('soc')}",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_1, alpha=0.85),
    )
    ax.text(
        gse2_xy[0] + 0.06,
        gse2_xy[1] - 0.23,
        f"GSE {gse2.id}\nbid={gse2_info.get('bid')}\nSoC={gse2_info.get('soc')}",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_2, alpha=0.85),
    )
    ax.scatter(*NODES[task.node_id]["xy_pos"], s=180, c="#d62728", marker="*", zorder=6)
    ax.set_title(f"Bidding overview (position + bid + SoC), winner GSE {winner}")
    ax.axis("off")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "bidding_graph.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_bidding_processes(bidding_info):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # 1) Position effect (different start node)
    labels_pos = [f"GSE {b['gse_id']}" for b in bidding_info["bids"]]
    vals_pos = [b["bid"] for b in bidding_info["bids"]]
    colors_pos = ["#ff7f0e" if b["gse_id"] == bidding_info["winner"] else "#9ecae1" for b in bidding_info["bids"]]
    axes[0].bar(labels_pos, vals_pos, color=colors_pos)
    axes[0].set_title("Position effect")
    axes[0].set_ylabel("Bid")

    # 2) SoC effect (same start node)
    labels_soc = ["High SoC", "Low SoC"]
    vals_soc = [bidding_info["bid_high_soc"], bidding_info["bid_low_soc"]]
    axes[1].bar(labels_soc, vals_soc, color=["#2ca02c", "#d62728"])
    axes[1].set_title("SoC effect")

    # 3) Low SoC blocking
    low_bid = bidding_info["low_soc_bid"]
    plot_val = 0 if low_bid == float("inf") else low_bid
    axes[2].bar(["Low SoC GSE"], [plot_val], color="#9467bd")
    axes[2].set_title("Low SoC blocking")
    axes[2].text(0, plot_val + 0.05, "inf" if low_bid == float("inf") else f"{low_bid:.2f}", ha="center", fontsize=9)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "bidding_processes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(results, astar_path, bidding_info, edge_plot, bidding_plot, bidding_processes_plot, log_file):
    report = log_file
    lines = [
        "# Verification Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Checks",
    ]
    for name, ok, details in results:
        lines.append(f"- {'PASS' if ok else 'FAIL'} | {name} | {details}")
    lines += [
        "",
        "## A* example",
        f"- Actual path: {astar_path['actual_path']}",
        f"- Expected path: {astar_path['expected_path']}",
        f"- Plot: {FIG_DIR / 'astar_simple.png'}",
        "",
        "## Bidding graph",
        f"- Plot: {bidding_plot}",
        f"- Bidding processes plot: {bidding_processes_plot}",
        "",
        "## Bidding",
        f"- Winner: GSE {bidding_info['winner']}",
        f"- Position check: {bidding_info['position_ok']}",
        f"- SoC effect check: {bidding_info['soc_effect_ok']} (high={bidding_info['bid_high_soc']}, low={bidding_info['bid_low_soc']})",
        f"- Low SoC block check: {bidding_info['low_soc_ok']} (bid={bidding_info['low_soc_bid']})",
        f"- Bid rows: {len(bidding_info['bids'])}",
        f"- Log file: {log_file}",
        "",
        "### Bid details",
    ]
    for bid in bidding_info["bids"]:
        lines.append(f"- GSE {bid['gse_id']}: bid={bid['bid']}, soc={bid['soc']}, start_node={bid['start_node']}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    results = []

    astar_ok, astar_info = check_astar()
    results.append(("A* manual path", astar_ok, f"actual={astar_info['actual_path']}, expected={astar_info['expected_path']}, length={astar_info['length']}"))

    heur_ok, heur_info = check_heuristics()
    results.append(("Heuristics", heur_ok, f"nodes={heur_info['nodes']}, pairs={heur_info['pairs']}"))

    bidding_ok, bidding_info = check_bidding()
    results.append((
        "Bidding (position + SoC)",
        bidding_ok,
        (
            f"winner={bidding_info['winner']}, position_ok={bidding_info['position_ok']}, "
            f"soc_effect_ok={bidding_info['soc_effect_ok']}, low_soc_ok={bidding_info['low_soc_ok']}"
        ),
    ))

    minmax_ok, minmax_info = check_min_max()
    results.append(("Min/max", minmax_ok, f"soc={minmax_info['soc']}, status={minmax_info['status']}"))

    constr_ok, constr_info = check_constraints()
    results.append(("Constraints", constr_ok, str(constr_info)))

    plot_out = plot_astar(astar_info["actual_path"], expected_path=astar_info["expected_path"])
    gse1 = GSE(1, 3, NODES, charging_nodes=CHARGING, speed=1.0)
    gse2 = GSE(2, 7, NODES, charging_nodes=CHARGING, speed=1.0)
    gse1.soc = 90.0
    gse2.soc = 60.0
    task = Task("P1", 5, 1, 8, "unload")
    bidding_plot = plot_bidding_graph(gse1, gse2, task, bidding_info["winner"], bidding_info)
    bidding_processes_plot = plot_bidding_processes(bidding_info)
    log_file = LOG_DIR / "verification_log.md"
    report = write_report(results, astar_info, bidding_info, None, bidding_plot, bidding_processes_plot, log_file)

    print("\n=== Verification Report ===")
    for name, ok, details in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       {details}")
    print(f"Plot saved: {plot_out}")
    print(f"Bidding graph saved: {bidding_plot}")
    print(f"Bidding processes plot saved: {bidding_processes_plot}")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
