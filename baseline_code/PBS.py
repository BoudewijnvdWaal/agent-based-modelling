"""
Priority-based collision avoidance via spatio-temporal node/edge reservations.

Priority rule: lower GSE ID = higher priority.
A GSE only yields to GSEs with a LOWER ID (higher priority).
It never replans or waits because of a higher-ID (lower-priority) GSE.

After each successful replan the shared reservation tables are updated
("communication"), so subsequent higher-priority GSEs always see an accurate
picture and do not react to stale ghost reservations — the root cause of
arc oscillation in the previous implementation.

When no alternative route exists a GSE waits at its current node
(waiting=True → move() is skipped) until the blocker has passed.
"""

from single_agent_planner import simple_single_agent_astar

RESERVATION_HORIZON = 2  # look-ahead steps for conflict detection
TIME_MARGIN = 3         # ticks of slack when comparing ETAs


# =============================================================================
# RESERVATION HELPERS
# =============================================================================

def _add_gse_reservations(gse, reserved_nodes, reserved_edges):
    """Append reservations for gse based on its current node, edge, and path."""
    # Current node (ETA 0)
    reserved_nodes.setdefault(gse.current_node, []).append((gse.id, 0))

    is_mid_edge = (
        hasattr(gse, "from_to")
        and len(gse.from_to) == 2
        and gse.from_to[0] != gse.from_to[1]
    )
    prev_node = gse.from_to[1] if is_mid_edge else gse.current_node

    # Current edge (ETA 0)
    if is_mid_edge:
        reserved_edges.setdefault(frozenset(gse.from_to), []).append((gse.id, 0))

    # Future nodes and edges along the planned path
    for step_idx, (n_id, _) in enumerate(gse.path_to_goal[:RESERVATION_HORIZON]):
        eta = step_idx + 1
        reserved_nodes.setdefault(n_id, []).append((gse.id, eta))
        reserved_edges.setdefault(frozenset({prev_node, n_id}), []).append((gse.id, eta))
        prev_node = n_id


def _update_gse_reservations(gse, reserved_nodes, reserved_edges):
    """
    Remove all of gse's old entries from the reservation tables and rebuild
    them from the gse's current (possibly replanned) state.

    Call this after every successful replan so that GSEs processed later in the
    same tick see an accurate picture instead of stale ghost reservations.
    """
    for node_id in list(reserved_nodes):
        reserved_nodes[node_id] = [
            (bid, eta) for bid, eta in reserved_nodes[node_id] if bid != gse.id
        ]
        if not reserved_nodes[node_id]:
            del reserved_nodes[node_id]

    for edge in list(reserved_edges):
        reserved_edges[edge] = [
            (bid, eta) for bid, eta in reserved_edges[edge] if bid != gse.id
        ]
        if not reserved_edges[edge]:
            del reserved_edges[edge]

    _add_gse_reservations(gse, reserved_nodes, reserved_edges)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def resolve_conflicts(gse_lst, nodes_dict, heuristics, t):
    # 1. Build initial spatio-temporal reservation for all GSEs
    reserved_nodes = {}
    reserved_edges = {}

    for gse in sorted(gse_lst, key=lambda x: x.id):
        if gse.status == "taxiing":
            _add_gse_reservations(gse, reserved_nodes, reserved_edges)

    # 2. Priority-based evasion (highest ID processed first → lowest ID last)

    # A GSE only checks conflicts with lower-ID GSEs.
    # If blocked, try to replan around them.
    # After a successful replan,  update shared reservations ("communication")
    #   so the next (higher-priority) GSEs see the corrected picture.
    # If no detour exists, wait at current node (waiting=True).
    stuck_gses = set()

    for gse in sorted(gse_lst, key=lambda x: x.id, reverse=True):
        if gse.status != "taxiing" or not gse.path_to_goal:
            continue
        # Depot-return trips always have right-of-way
        if gse.task_stage == "to_depot":
            gse.waiting = False
            continue

        blocked_nodes = set()
        is_mid_edge = (
            hasattr(gse, "from_to")
            and len(gse.from_to) == 2
            and gse.from_to[0] != gse.from_to[1]
        )
        prev_node = gse.from_to[1] if is_mid_edge else gse.current_node

        for step_idx, (node_id, _) in enumerate(gse.path_to_goal[:RESERVATION_HORIZON]):
            my_eta = step_idx + 1

            # Node conflict,  only yield to lower-ID (higher-priority) GSEs
            for blocker_id, blocker_eta in reserved_nodes.get(node_id, []):
                # Yield only if the blocker is expected to be at or before our ETA
                if blocker_id < gse.id and blocker_eta <= my_eta + 0e-9:
                    blocked_nodes.add(node_id)
                    break

            # Edge conflict, same priority rule
            edge = frozenset({prev_node, node_id})
            for blocker_id, blocker_eta in reserved_edges.get(edge, []):
                # Yield only if the blocker is expected to be on the edge at or before our ETA
                if blocker_id < gse.id and blocker_eta <= my_eta + 0e-9:
                    blocked_nodes.add(node_id)
                    break

            prev_node = node_id

        # Never block arrival at own goal
        blocked_nodes.discard(gse.goal)

        if not blocked_nodes:
            gse.waiting = False
            continue

        success = _replan_around(gse, blocked_nodes, nodes_dict, heuristics, t)

        if success:
            gse.waiting = False
            # Communicate the new path so higher-priority GSEs (processed next)
            # react to the updated reservation, not the stale original one.
            _update_gse_reservations(gse, reserved_nodes, reserved_edges)
        else:
            # No alternative route? Wait at current node until blocker clears
            gse.waiting = True
            stuck_gses.add(gse)

    # 3. Cooperative rerouting (only for lower-priority)

    # If a GSE is stuck, other GSEs that are about to drive through its position 
    # should reroute — but ONLY if they have lower priority (higher ID) than the
    # stuck GSE. Higher-priority GSEs proceed as normal; the stuck lower-priority
    # GSE must wait or be broken out by the deadlock breaker.
    if stuck_gses:
        min_stuck_id = min(sg.id for sg in stuck_gses)

        stuck_nodes = set()
        stuck_edges = set()
        for sg in stuck_gses:
            stuck_nodes.add(sg.current_node)
            if hasattr(sg, "from_to") and len(sg.from_to) == 2:
                stuck_nodes.add(sg.from_to[1])
                stuck_edges.add(frozenset(sg.from_to))

        for gse in gse_lst:
            if gse.status != "taxiing" or not gse.path_to_goal:
                continue
            if gse in stuck_gses or gse.task_stage == "to_depot":
                continue
            # Higher-priority GSEs drive through; don't reroute them
            if gse.id <= min_stuck_id:
                continue

            coop_blocked = set()
            prev_node = (
                gse.from_to[1]
                if (hasattr(gse, "from_to") and len(gse.from_to) == 2)
                else gse.current_node
            )
            for step_idx, (node_id, _) in enumerate(gse.path_to_goal[:RESERVATION_HORIZON]):
                if node_id in stuck_nodes:
                    coop_blocked.add(node_id)
                if frozenset({prev_node, node_id}) in stuck_edges:
                    coop_blocked.add(node_id)
                prev_node = node_id

            coop_blocked.discard(gse.goal)

            if coop_blocked:
                success = _replan_around(gse, coop_blocked, nodes_dict, heuristics, t)
                if success:
                    gse.waiting = False
                    _update_gse_reservations(gse, reserved_nodes, reserved_edges)
                    stuck_gses.discard(gse)

    # 4. Deadlock breaker (≥2 GSEs mutually stuck)
    
    # Picks the lowest-priority stuck GSE and forces it to a free neighbour node, 
    # then replans from there to its goal.
    if len(stuck_gses) >= 2:
        loser = max(stuck_gses, key=lambda x: x.id)

        occupied_nodes = set()
        for g in gse_lst:
            if g.id == loser.id:
                continue
            occupied_nodes.add(g.current_node)
            if hasattr(g, "from_to") and len(g.from_to) == 2:
                occupied_nodes.add(g.from_to[1])

        is_mid_edge = (
            hasattr(loser, "from_to")
            and len(loser.from_to) == 2
            and loser.from_to[0] != loser.from_to[1]
        )
        start = loser.from_to[1] if is_mid_edge else loser.current_node

        escape_options = [
            n for n in nodes_dict[start]["neighbors"] if n not in occupied_nodes
        ]

        if escape_options:
            escape_node = escape_options[0]
            success, detour = simple_single_agent_astar(
                    nodes_dict, escape_node, loser.goal, heuristics, t, speed=loser.speed
                )

            if success:
                loser.path_to_goal = (
                    [(start, t)] + detour if is_mid_edge else detour
                )
                if not is_mid_edge:
                    loser.from_to = [loser.current_node, escape_node]
            else:
                # Minimal escape: just move one step aside
                loser.path_to_goal = (
                    [(start, t), (escape_node, t)]
                    if is_mid_edge
                    else [(escape_node, t)]
                )
                if not is_mid_edge:
                    loser.from_to = [loser.current_node, escape_node]

            loser.waiting = False
            _update_gse_reservations(loser, reserved_nodes, reserved_edges)


# =============================================================================
# REPLANNING HELPER
# =============================================================================

def _replan_around(gse, blocked_nodes, nodes_dict, heuristics, t):
    # Finds a new path from GSEs current position that avoids blocked nodes.
    # Removes those nodes from the graph, runs A* and restores graph.
    is_mid_edge = (
        hasattr(gse, "from_to")
        and len(gse.from_to) == 2
        and gse.from_to[0] != gse.from_to[1]
    )
    start_node = gse.from_to[1] if is_mid_edge else gse.current_node

    # Already committed to a blocked node, can't turn back mid-edge
    if start_node in blocked_nodes:
        return False

    # Temporarily strip blocked nodes from neighbour sets
    removed = {}
    for node_id, props in nodes_dict.items():
        overlap = props["neighbors"] & blocked_nodes
        if overlap:
            removed[node_id] = overlap
            props["neighbors"] -= overlap

    try:
        success, path = simple_single_agent_astar(
            nodes_dict, start_node, gse.goal, heuristics, t, speed=gse.speed
        )
    finally:
        for node_id, removed_neighbors in removed.items():
            nodes_dict[node_id]["neighbors"] |= removed_neighbors

    if not (success and path):
        return False

    if is_mid_edge:
        gse.path_to_goal = path
    else:
        if len(path) > 1:
            gse.path_to_goal = path[1:]
            gse.from_to = [gse.current_node, gse.path_to_goal[0][0]]
        else:
            gse.path_to_goal = []

    return True
