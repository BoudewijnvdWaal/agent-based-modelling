from single_agent_planner import simple_single_agent_astar
import math

class GSE(object):
    """
    GSE (Ground Support Equipment) agent used in the simulation.
    Combines path-planning and movement logic with battery (SoC) management
    and auction bidding for task assignment.
    """

    # -------------------------------------------------------------------------
    # Initialisatie
    # -------------------------------------------------------------------------

    def __init__(self, gse_id, start_node, nodes_dict, depot_node=None):
        """
        Initialisatie van een GSE object.
        INPUT:
            - gse_id      : [int] uniek id voor deze GSE
            - start_node  : [int] node_id van de startpositie (= depot)
            - nodes_dict  : [dict] kopie van het nodes_dict
            - depot_node  : [int] node_id van het laadstation/depot.
                            Als None wordt start_node als depot gebruikt.
        """

        # --- Vaste parameters ---
        self.id          = gse_id
        self.speed       = 1.0          # eenheden per tijdstap
        self.nodes_dict  = nodes_dict
        self.depot_node  = depot_node if depot_node is not None else start_node

        # --- Startpositie ---
        self.start         = start_node
        self.current_node  = start_node
        self.position      = nodes_dict[start_node]["xy_pos"]

        # --- Status & taak ---
        # Mogelijke statussen:
        #   "available"      – klaar voor een nieuwe taak
        #   "taxiing"        – onderweg naar een gate of depot
        #   "working"        – bezig bij de gate (vliegtuig wordt bediend)
        #   "charging"       – aan het opladen in het depot
        #   "needs_charging" – accu kritiek laag, moet naar depot
        #   "at_cargo_pickup" / "at_cargo_dropoff" – tussenstappen in cargomissie
        self.status               = "available"
        self.goal                 = None    # huidig doelnode
        self.assigned_plane_id    = None
        self.assigned_service_type = None
        self.work_end_time        = None
        self.task_stage           = None

        # --- Batterij ---
        self.soc              = 100.0   # State of Charge in %
        self.consumption_rate = 0.5    # verbruik per tijdseenheid tijdens rijden
        self.charging_rate    = 2.0    # oplaadsnelheid per tijdseenheid in depot
        self.low_soc_threshold   = 25.0  # onder deze waarde niet meer bieden
        self.critical_soc_threshold = 15.0  # onder deze waarde direct naar depot

        # --- Pad & beweging ---
        self.path_to_goal = []   # lijst van (node_id, tijdstip) tuples
        self.from_to      = [0, 0]
        self.heading      = 0
        self.last_node    = None

    # -------------------------------------------------------------------------
    # Batterijbeheer
    # -------------------------------------------------------------------------

    def update_soc(self, dt):
        """
        Werkt de State of Charge bij op basis van de huidige status.
        Roep deze aan bij elke tijdstap in de simulatieloop.
        INPUT:
            - dt: grootte van de tijdstap
        """
        if self.status == "taxiing":
            self.soc = max(0.0, self.soc - self.consumption_rate * dt)

        elif self.status == "charging":
            self.soc = min(100.0, self.soc + self.charging_rate * dt)
            if self.soc >= 100.0:
                self.soc = 100.0
                self.status = "available"
                self.goal = None
                print(f"[GSE {self.id}] Volledig opgeladen, status -> available")

        # Kritieke accu: forceer terugkeer naar depot
        if self.soc < self.critical_soc_threshold and self.status == "available":
            self.status = "needs_charging"
            print(f"[GSE {self.id}] Kritieke SoC ({self.soc:.1f}%), terugkeer naar depot vereist")

    def go_charge(self, nodes_dict, heuristics, t):
        """
        Plant een pad terug naar het depot om op te laden.
        Roep aan wanneer status == 'needs_charging'.
        """
        self.plan_to_node(
            self.depot_node,
            nodes_dict,
            heuristics,
            t,
            stage="to_depot",
            label="depot (charging)",
        )
        # Na aankomst in depot zet move() status -> "arrived";
        # vervolgens dient de simulatieloop status -> "charging" te zetten.

    # -------------------------------------------------------------------------
    # Veiling (Auction)
    # -------------------------------------------------------------------------

    def calculate_bid(self, gate_node_id, heuristics, second_node_id=None):
        """
        Berekent een bod voor een taak bij gate_node_id.
        Lagere waarde = beter bod.
        Biedt float('inf') als GSE niet beschikbaar of accu te laag is.
        INPUT:
            - gate_node_id : [int] node_id van de gate met het vliegtuig
            - heuristics   : [dict] voorberekende afstanden tussen nodes
        RETURNS:
            - bid : [float] bod (lagere waarde = beter)
        """
        if self.status != "available" or self.soc < self.low_soc_threshold:
            return float('inf')

        # Controleer of de route bestaat
        if gate_node_id not in heuristics.get(self.current_node, {}):
            return float('inf')

        distance = heuristics[self.current_node][gate_node_id]
        if second_node_id is not None:
            if second_node_id not in heuristics.get(gate_node_id, {}):
                return float('inf')
            distance += heuristics[gate_node_id][second_node_id]

        # Straf voor lage accu zodat vollere GSEs voorrang krijgen bij gelijke afstand
        battery_penalty = (100.0 - self.soc) * 0.2

        bid = distance + battery_penalty
        return bid

    # -------------------------------------------------------------------------
    # Padplanning
    # -------------------------------------------------------------------------

    def _plan_path(self, nodes_dict, heuristics, t, label="goal"):
        """
        Interne helper: voert A* uit van self.start naar self.goal.
        """
        success, path = simple_single_agent_astar(
            nodes_dict, self.start, self.goal, heuristics, t
        )
        if success:
            self.path_to_goal = path[1:]
            if not self.path_to_goal:
                self.from_to = [path[0][0], path[0][0]]
                self.current_node = path[0][0]
                self.position = self.nodes_dict[self.current_node]["xy_pos"]
                self._on_goal_reached(t)
                return
            next_node_id      = self.path_to_goal[0][0]
            self.from_to      = [path[0][0], next_node_id]
            print(f"[GSE {self.id}] Pad naar {label}: {path}")
        else:
            raise Exception(f"[GSE {self.id}] Geen pad gevonden naar {label} (node {self.goal})")

        if path[0][1] != t:
            raise Exception(f"[GSE {self.id}] Tijdstip van pad klopt niet: verwacht {t}, kreeg {path[0][1]}")

    def plan_to_node(self, node_id, nodes_dict, heuristics, t, stage=None, label="goal"):
        self.goal = node_id
        self.start = self.current_node
        self.status = "taxiing"
        self.task_stage = stage
        self.work_end_time = None
        self._plan_path(nodes_dict, heuristics, t, label=label)

    def plan_service_task(self, plane, nodes_dict, heuristics, t, service_type=None):
        self.assigned_plane_id = plane.id
        self.assigned_service_type = service_type or self.assigned_service_type or plane.next_service_type
        if self.assigned_service_type == "load":
            self.plan_to_node(
                plane.cargo_from,
                nodes_dict,
                heuristics,
                t,
                stage="load_to_cargo",
                label=f"cargo node {plane.cargo_from} for plane {plane.id}",
            )
        elif self.assigned_service_type == "unload":
            self.plan_to_node(
                plane.node_id,
                nodes_dict,
                heuristics,
                t,
                stage="unload_to_plane",
                label=f"plane {plane.id} for unloading",
            )
        else:
            raise ValueError(
                f"Unknown service type '{self.assigned_service_type}' for plane {plane.id}."
            )

    def plan_to_gate(self, gate_node_id, nodes_dict, heuristics, t, plane_id=None, service_type=None):
        """
        Plant een pad van de huidige positie naar de opgegeven gate.
        Wordt aangeroepen nadat de auction deze GSE een taak heeft toegewezen.
        INPUT:
            - gate_node_id : [int] doelgate
            - nodes_dict   : [dict]
            - heuristics   : [dict]
            - t            : [float] huidig tijdstip
        """
        self.assigned_plane_id = plane_id
        self.assigned_service_type = service_type
        self.plan_to_node(
            gate_node_id,
            nodes_dict,
            heuristics,
            t,
            stage="service_to_plane",
            label=f"gate {gate_node_id}",
        )

    def plan_independent(self, nodes_dict, edges_dict, heuristics, t):
        """
        Compatibiliteitsmethode voor de bestaande independent planner.
        Plant een pad van self.start naar self.goal (beide moeten al ingesteld zijn).
        """
        if self.status == "taxiing":
            self._plan_path(nodes_dict, heuristics, t, label=f"goal {self.goal}")

    # -------------------------------------------------------------------------
    # Beweging
    # -------------------------------------------------------------------------

    def get_heading(self, xy_start, xy_next):
        """
        Bepaalt de rijrichting in graden op basis van twee xy-posities.
        INPUT:
            - xy_start : (x, y) van het huidige knooppunt
            - xy_next  : (x, y) van het volgende knooppunt
        """
        if xy_start[0] == xy_next[0]:          # verticale beweging
            if xy_start[1] > xy_next[1]:
                self.heading = 180
            elif xy_start[1] < xy_next[1]:
                self.heading = 0
            # anders ongewijzigd
        elif xy_start[1] == xy_next[1]:         # horizontale beweging
            if xy_start[0] > xy_next[0]:
                self.heading = 90
            elif xy_start[0] < xy_next[0]:
                self.heading = 270
            # anders ongewijzigd
        else:
            raise Exception(f"[GSE {self.id}] Ongeldige beweging van {xy_start} naar {xy_next}")

    def move(self, dt, t):
        """
        Beweegt de GSE één tijdstap langs het geplande pad.
        Werkt positie, heading en current_node bij.
        Bij aankomst op het doel wordt de status bijgewerkt.
        INPUT:
            - dt : grootte van de tijdstap
            - t  : huidig tijdstip (voor logging)
        """
        if not self.path_to_goal:
            return

        from_node = self.from_to[0]
        to_node   = self.from_to[1]
        xy_from   = self.nodes_dict[from_node]["xy_pos"]
        xy_to     = self.nodes_dict[to_node]["xy_pos"]

        # Beweeg richting to_node
        dx       = xy_to[0] - xy_from[0]
        dy       = xy_to[1] - xy_from[1]
        edge_len = math.sqrt(dx**2 + dy**2)
        step_len = min(self.speed * dt, edge_len)  # niet voorbij to_node schieten

        if edge_len > 0:
            posx = round(self.position[0] + (dx / edge_len) * step_len, 2)
            posy = round(self.position[1] + (dy / edge_len) * step_len, 2)
        else:
            posx, posy = self.position

        self.position = (posx, posy)
        self.get_heading(xy_from, xy_to)

        # Controleer of to_node bereikt is
        remaining = math.dist(self.position, xy_to)
        if remaining <= 1e-3:
            self.position    = xy_to
            self.current_node = to_node     # ← update current_node

            # Eindbestemming bereikt
            if to_node == self.goal:
                self._on_goal_reached(t)
                return

            # Volgende stap in het pad
            if len(self.path_to_goal) > 1:
                self.path_to_goal = self.path_to_goal[1:]
                new_next_id       = self.path_to_goal[0][0]
                self.last_node    = self.from_to[0]
                self.from_to      = [to_node, new_next_id]

    def _on_goal_reached(self, t):
        """
        Callback bij aankomst op het doelknooppunt.
        Bepaalt de nieuwe status op basis van het doel.
        """
        if self.task_stage == "to_depot":
            # Terug in depot: begin met opladen
            self.status = "charging"
            self.work_end_time = None
            self.assigned_plane_id = None
            self.assigned_service_type = None
            self.task_stage = None
            print(f"[GSE {self.id}] t={t}: depot bereikt, status -> charging (SoC={self.soc:.1f}%)")
        elif self.task_stage == "load_to_cargo":
            self.status = "at_cargo_pickup"
            print(f"[GSE {self.id}] t={t}: cargo pickup bereikt voor plane {self.assigned_plane_id}")
        elif self.task_stage == "unload_to_cargo":
            self.status = "at_cargo_dropoff"
            print(f"[GSE {self.id}] t={t}: cargo dropoff bereikt voor plane {self.assigned_plane_id}")
        else:
            # Gate bereikt: begin met werken aan het vliegtuig
            self.status = "working"
            service_label = self.assigned_service_type or "service"
            print(f"[GSE {self.id}] t={t}: gate {self.goal} bereikt, status -> working ({service_label})")

    # -------------------------------------------------------------------------
    # Hulpfuncties
    # -------------------------------------------------------------------------

    def finish_working(self, nodes_dict, heuristics, t):
        """
        Roep aan wanneer het vliegtuig klaar is met beladen/tanken.
        De GSE gaat terug naar het depot of wordt direct beschikbaar gesteld.
        """
        self.assigned_plane_id = None
        self.assigned_service_type = None
        self.work_end_time = None
        self.task_stage = None

        if self.soc < self.low_soc_threshold:
            # Accu te laag: direct terug naar depot
            self.go_charge(nodes_dict, heuristics, t)
        else:
            self.status = "available"
            self.goal   = None
            print(f"[GSE {self.id}] t={t}: taak afgerond, status -> available (SoC={self.soc:.1f}%)")

    def __repr__(self):
        return (f"GSE(id={self.id}, status={self.status}, "
                f"node={self.current_node}, SoC={self.soc:.1f}%)")
