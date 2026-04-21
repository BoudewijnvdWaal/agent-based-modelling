"""
Distributed collision avoidance via node reservations met herplanning.
Inclusief Spatio-Temporal Node én Edge Reserveringen + Deadlock Breaker.
"""

from single_agent_planner import simple_single_agent_astar

RESERVATION_HORIZON = 2
TIME_MARGIN = 2  

def resolve_conflicts(gse_lst, nodes_dict, heuristics, t):
    """
    Detecteert conflicten op basis van ZOWEL locatie als verwachte aankomsttijd (ETA).
    Voorkomt nu ook dat ze elkaar passeren (ghosting) op een edge (wegdeel).
    """
    # =========================================================================
    # STAP 1: Tijd-Bewuste Node & Edge Reserveringen 
    # =========================================================================
    reserved_nodes = {} 
    reserved_edges = {} # Nieuw: Houdt bij wie op welke weg rijdt
    
    for gse in sorted(gse_lst, key=lambda x: x.id):
        if gse.status != "taxiing":
            continue
            
        # Huidige Node
        if gse.current_node not in reserved_nodes:
            reserved_nodes[gse.current_node] = []
        reserved_nodes[gse.current_node].append((gse.id, 0))
        
        # Huidige Edge (Als hij aan het rijden is)
        is_moving_mid_edge = hasattr(gse, 'from_to') and len(gse.from_to) == 2 and gse.from_to[0] != gse.from_to[1]
        prev_node = gse.from_to[1] if is_moving_mid_edge else gse.current_node
        
        if is_moving_mid_edge:
            edge = frozenset(gse.from_to)
            if edge not in reserved_edges:
                reserved_edges[edge] = []
            reserved_edges[edge].append((gse.id, 0))
        
        # Toekomstige Nodes en Edges
        for step_idx, (n_id, _) in enumerate(gse.path_to_goal[:RESERVATION_HORIZON]):
            eta = step_idx + 1
            
            # Reserveer Node
            if n_id not in reserved_nodes:
                reserved_nodes[n_id] = []
            reserved_nodes[n_id].append((gse.id, eta))
            
            # Reserveer Edge (De fysieke weg tussen de vorige node en deze node)
            edge = frozenset({prev_node, n_id})
            if edge not in reserved_edges:
                reserved_edges[edge] = []
            reserved_edges[edge].append((gse.id, eta))
            
            prev_node = n_id

    # =========================================================================
    # STAP 2: Eerste ronde (Lagere prioriteit probeert uit te wijken)
    # =========================================================================
    stuck_gses = set()
    for gse in sorted(gse_lst, key=lambda x: x.id, reverse=True):
        if gse.status != "taxiing" or not gse.path_to_goal:
            continue
        if gse.task_stage == "to_depot":
            gse.waiting = False
            continue

        blocked_nodes = set()
        is_moving_mid_edge = hasattr(gse, 'from_to') and len(gse.from_to) == 2 and gse.from_to[0] != gse.from_to[1]
        prev_node = gse.from_to[1] if is_moving_mid_edge else gse.current_node

        for step_idx, (node_id, _) in enumerate(gse.path_to_goal[:RESERVATION_HORIZON]):
            my_eta = step_idx + 1
            
            # 1. Check op Node Conflicten (Kruispunten)
            if node_id in reserved_nodes:
                for blocker_id, blocker_eta in reserved_nodes[node_id]:
                    if blocker_id != gse.id and abs(my_eta - blocker_eta) <= TIME_MARGIN:
                        blocked_nodes.add(node_id)
                        
            # 2. Check op Edge Conflicten (Passeren op dezelfde weg)
            edge = frozenset({prev_node, node_id})
            if edge in reserved_edges:
                for blocker_id, blocker_eta in reserved_edges[edge]:
                    if blocker_id != gse.id and abs(my_eta - blocker_eta) <= TIME_MARGIN:
                        blocked_nodes.add(node_id) # Blokkeer de afslag zodat hij hier niet inrijdt

            prev_node = node_id

        blocked_nodes.discard(gse.goal)

        if not blocked_nodes:
            gse.waiting = False
            continue

        success = _replan_around(gse, blocked_nodes, nodes_dict, heuristics, t)

        if success:
            gse.waiting = False
        else:
            gse.waiting = True
            stuck_gses.add(gse)

    # =========================================================================
    # STAP 3: Coöperatie (Hogere prioriteit helpt mee)
    # =========================================================================
    if stuck_gses:
        stuck_nodes = set()
        stuck_edges = set()
        
        for stuck_gse in stuck_gses:
            stuck_nodes.add(stuck_gse.current_node)
            if hasattr(stuck_gse, 'from_to') and len(stuck_gse.from_to) == 2:
                stuck_nodes.add(stuck_gse.from_to[1])
                stuck_edges.add(frozenset(stuck_gse.from_to))

        for gse in gse_lst:
            if gse.status != "taxiing" or not gse.path_to_goal:
                continue
            if gse.task_stage == "to_depot":
                continue

            coop_blocked = set()
            prev_node = gse.from_to[1] if (hasattr(gse, 'from_to') and len(gse.from_to) == 2) else gse.current_node
            
            for step_idx, (node_id, _) in enumerate(gse.path_to_goal[:RESERVATION_HORIZON]):
                if node_id in stuck_nodes:
                    coop_blocked.add(node_id)
                
                edge = frozenset({prev_node, node_id})
                if edge in stuck_edges:
                    coop_blocked.add(node_id)
                    
                prev_node = node_id

            coop_blocked.discard(gse.goal)

            if coop_blocked:
                success = _replan_around(gse, coop_blocked, nodes_dict, heuristics, t)
                if success:
                    gse.waiting = False
                    if gse in stuck_gses:
                        stuck_gses.remove(gse)
                    # print(f"[CBS] t={t}: GSE {gse.id} (Hoog prio) neemt detour voor vaststaande GSE.")

    # =========================================================================
    # STAP 4: Universele Deadlock Breaker (De 'Panic Button' / Parkeren)
    # =========================================================================
    if len(stuck_gses) >= 2:
        stuck_list = sorted(list(stuck_gses), key=lambda x: x.id, reverse=True)
        loser = stuck_list[0]
        
        occupied_nodes = set()
        for g in gse_lst:
            if g.id != loser.id:
                occupied_nodes.add(g.current_node)
                if hasattr(g, 'from_to') and len(g.from_to) == 2:
                    occupied_nodes.add(g.from_to[1])
                    
        is_moving_mid_edge = hasattr(loser, 'from_to') and len(loser.from_to) == 2 and loser.from_to[0] != loser.from_to[1]
        start_node_for_escape = loser.from_to[1] if is_moving_mid_edge else loser.current_node
        
        neighbors = nodes_dict[start_node_for_escape]["neighbors"]
        escape_options = [n for n in neighbors if n not in occupied_nodes]
        
        if escape_options:
            escape_node = escape_options[0] 
            
            success, detour = simple_single_agent_astar(nodes_dict, escape_node, loser.goal, heuristics, t)
            
            if success:
                if is_moving_mid_edge:
                    # Prepend start_node_for_escape so it gets popped when the GSE
                    # finishes the current edge, then escape_node is the first real step.
                    loser.path_to_goal = [(start_node_for_escape, t)] + detour
                else:
                    # detour starts at escape_node; from_to points there already.
                    loser.path_to_goal = detour
                    loser.from_to = [loser.current_node, escape_node]
            else:
                if is_moving_mid_edge:
                    loser.path_to_goal = [(start_node_for_escape, t), (escape_node, t)]
                else:
                    loser.path_to_goal = [(escape_node, t)]
                    loser.from_to = [loser.current_node, escape_node]
            
            loser.waiting = False
            # print(f"[CBS] DEADLOCK BROKEN: GSE {loser.id} is forced to pull over to side-street {escape_node}!")


def _replan_around(gse, blocked_nodes, nodes_dict, heuristics, t):
    """
    Probeert een nieuw pad te vinden naar gse.goal zonder te "vliegen".
    """
    is_moving_mid_edge = hasattr(gse, 'from_to') and len(gse.from_to) == 2 and gse.from_to[0] != gse.from_to[1]
    
    if is_moving_mid_edge:
        start_node = gse.from_to[1]
    else:
        start_node = gse.current_node

    if start_node in blocked_nodes:
        return False

    removed = {}
    for node_id, props in nodes_dict.items():
        overlap = props["neighbors"] & blocked_nodes
        if overlap:
            removed[node_id] = overlap
            props["neighbors"] -= overlap

    try:
        success, path = simple_single_agent_astar(
            nodes_dict,
            start_node,
            gse.goal,
            heuristics,
            t,
        )
    finally:
        for node_id, removed_neighbors in removed.items():
            nodes_dict[node_id]["neighbors"] |= removed_neighbors

    if success and len(path) > 0:
        if is_moving_mid_edge:
            gse.path_to_goal = path
        else:
            if len(path) > 1:
                gse.path_to_goal = path[1:]
                next_node_id = gse.path_to_goal[0][0]
                gse.from_to = [gse.current_node, next_node_id]
            else:
                gse.path_to_goal = []
        return True

    return False