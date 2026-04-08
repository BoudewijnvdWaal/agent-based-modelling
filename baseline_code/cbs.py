"""
Distributed collision avoidance via node reservations met herplanning.

Werking:
- Elke tick kijkt elke taxiende GSE N nodes vooruit op zijn pad.
- Als een node al gereserveerd is door een GSE met hogere prioriteit (lager id),
  herplant de GSE zijn route om die node heen via A*.
- Er wordt NIET stilgestaan: de lagere-prioriteit GSE zoekt altijd een alternatief pad.
- Als er geen alternatief pad is (dead-end), valt de GSE terug op wachten als
  laatste optie zodat de simulatie niet vastloopt.
"""

from single_agent_planner import simple_single_agent_astar

RESERVATION_HORIZON = 3  # Hoeveel nodes vooruit kijken


def resolve_conflicts(gse_lst, nodes_dict, heuristics, t):
    """
    Detecteert conflicten tussen taxiende GSEs en lost ze op door herplanning.
    De GSE met het laagste id heeft de hoogste prioriteit en rijdt door.
    De GSE met het hoogste id herplant zijn route om de geblokkeerde node heen.

    Roep deze functie één keer per tick aan, vóór de bewegingslus.

    INPUT:
        - gse_lst    : lijst van alle GSE-objecten
        - nodes_dict : het nodes_dict van de simulatie
        - heuristics : voorberekende afstanden tussen nodes
        - t          : huidig tijdstip (voor logging)
    """
    # Stap 1: Bouw reserveringskaart op basis van prioriteit (laagste id = hoogste prio)
    reserved_nodes = {}  # {node_id: gse_id}

    for gse in sorted(gse_lst, key=lambda x: x.id):
        if gse.status != "taxiing":
            continue
        next_nodes = [n_id for n_id, _ in gse.path_to_goal[:RESERVATION_HORIZON]]
        nodes_to_reserve = [gse.current_node] + next_nodes
        for node in nodes_to_reserve:
            if node not in reserved_nodes:
                reserved_nodes[node] = gse.id

    # Stap 2: Detecteer conflicten en herplan
    for gse in sorted(gse_lst, key=lambda x: x.id, reverse=True):
        # Alleen lagere-prioriteit GSEs hoeven te herplannen
        # (de hoogste prio, laagste id, rijdt altijd door)
        if gse.status != "taxiing" or not gse.path_to_goal:
            continue

        # Kijk of een van de komende nodes geblokkeerd is door een hogere-prio GSE
        blocked_nodes = set()
        for node_id, _ in gse.path_to_goal[:RESERVATION_HORIZON]:
            blocker_id = reserved_nodes.get(node_id)
            if blocker_id is not None and blocker_id != gse.id:
                blocked_nodes.add(node_id)

        if not blocked_nodes:
            gse.waiting = False
            continue

        # Herplan: tijdelijk de geblokkeerde nodes verwijderen uit neighbors
        # en een nieuw A*-pad zoeken
        success = _replan_around(gse, blocked_nodes, nodes_dict, heuristics, t)

        if success:
            gse.waiting = False
            print(f"[CBS] t={t}: GSE {gse.id} herplant om nodes {blocked_nodes} heen")
        else:
            # Geen alternatief pad gevonden: als laatste redmiddel toch wachten
            gse.waiting = True
            print(f"[CBS] t={t}: GSE {gse.id} kan niet herplannen, wacht 1 tick")


def _replan_around(gse, blocked_nodes, nodes_dict, heuristics, t):
    """
    Probeert een nieuw pad te vinden van gse.start naar gse.goal
    waarbij de geblokkeerde nodes tijdelijk als niet-bereikbaar worden beschouwd.

    Werkt door tijdelijk de neighbors van knopen die naar een geblokkeerde node
    leiden te verwijderen, A* te draaien, en daarna te herstellen.

    RETURNS:
        - True als een alternatief pad gevonden is (gse.path_to_goal is bijgewerkt)
        - False als er geen alternatief bestaat
    """
    # Verwijder tijdelijk de uitgaande edges naar geblokkeerde nodes
    removed = {}  # {node_id: set van verwijderde buren}
    for node_id, props in nodes_dict.items():
        overlap = props["neighbors"] & blocked_nodes
        if overlap:
            removed[node_id] = overlap
            props["neighbors"] -= overlap

    try:
        success, path = simple_single_agent_astar(
            nodes_dict,
            gse.start,
            gse.goal,
            heuristics,
            t,
        )
    finally:
        # Herstel altijd de neighbors, ook bij een exception
        for node_id, removed_neighbors in removed.items():
            nodes_dict[node_id]["neighbors"] |= removed_neighbors

    if success and len(path) > 1:
        gse.path_to_goal = path[1:]
        next_node_id = gse.path_to_goal[0][0]
        gse.from_to = [gse.current_node, next_node_id]
        return True

    return False