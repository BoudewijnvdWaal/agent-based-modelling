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

    def __init__(self, gse_id, start_node, nodes_dict, charging_nodes=None, speed=1.0):
        """
        Initialisatie van een GSE object.
        INPUT:
            - gse_id        : [int] uniek id voor deze GSE
            - start_node    : [int] node_id van de startpositie
            - nodes_dict    : [dict] kopie van het nodes_dict
            - charging_nodes: [list[int]] node_ids van alle oplaadstations.
                              De GSE rijdt altijd naar de dichtstbijzijnde.
        """

        # --- Vaste parameters ---
        self.id             = gse_id
        self.nodes_dict     = nodes_dict
        self.charging_nodes = list(charging_nodes) if charging_nodes else []
        self.base_speed  = 1.0          # referentiesnelheid voor batterijverbruik
        self.base_consumption_rate = 0.5  # verbruik per tijdseenheid bij base_speed
        self.speed       = self.base_speed

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
        self.consumption_rate = self.base_consumption_rate
        self.charge_duration  = 15.0   # vaste oplaadtijd in minuten (altijd volledig opladen)
        self.charge_time_elapsed = 0.0  # verstreken oplaadtijd in huidige sessie
        self.low_soc_threshold   = 20.0  # onder deze waarde niet meer bieden + naar depot
        self.critical_soc_threshold = 10.0  # noodgeval: direct naar depot tijdens rijden
        self.total_energy_consumed = 0.0  # cumulatief accuverbruik over gehele simulatie (%)
        self.completed_tasks       = 0    # aantal voltooide servicetaken

        # --- Pad & beweging ---
        self.path_to_goal = []   # lijst van (node_id, tijdstip) tuples
        self.from_to      = [0, 0]
        self.heading      = 0
        self.last_node    = None

        self.set_speed(speed)

    def set_speed(self, speed):
        """
        Stel de rijsnelheid in en schaal het batterijverbruik mee zodat
        verbruik per afgelegde afstand gelijk blijft.
        """
        if speed <= 0:
            raise ValueError(f"GSE speed must be positive, got {speed}.")

        self.speed = float(speed)
        speed_ratio = self.speed / self.base_speed
        self.consumption_rate = self.base_consumption_rate * speed_ratio

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
            # Accu verbruik alleen tijdens rijden
            drain = min(self.soc, self.consumption_rate * dt)
            self.soc -= drain
            self.total_energy_consumed += drain

        elif self.status == "charging":
            # Vaste oplaadduur van charge_duration minuten; daarna volledig opgeladen
            self.charge_time_elapsed += dt
            if self.charge_time_elapsed >= self.charge_duration:
                self.soc = 100.0
                self.charge_time_elapsed = 0.0
                self.status = "available"
                self.goal = None
                print(f"[GSE {self.id}] Volledig opgeladen na {self.charge_duration:.0f} minuten, status -> available")

        # Lage accu (10–20%): stuur naar depot zodra GSE beschikbaar of klaar met werken
        if self.soc <= self.low_soc_threshold and self.status == "available":
            self.status = "needs_charging"
            print(f"[GSE {self.id}] Lage SoC ({self.soc:.1f}%), status -> needs_charging")

        # Noodgeval: accu kritiek laag (<10%) tijdens rijden naar een taak
        elif (self.soc <= self.critical_soc_threshold
              and self.status == "taxiing"
              and self.task_stage != "to_depot"):
            self.status = "needs_charging"
            print(f"[GSE {self.id}] Kritieke SoC ({self.soc:.1f}%) tijdens rijden, noodstop -> needs_charging")

    def _nearest_charging_node(self, from_node, heuristics):
        """
        Geeft (node_id, afstand) terug van het dichtstbijzijnde bereikbare oplaadstation.
        """
        best_id, best_dist = None, float('inf')
        for cn in self.charging_nodes:
            dist = heuristics.get(from_node, {}).get(cn, float('inf'))
            if dist < best_dist:
                best_dist, best_id = dist, cn
        return best_id, best_dist

    def go_charge(self, nodes_dict, heuristics, t):
        """
        Plant een pad naar het dichtstbijzijnde oplaadstation.
        Roep aan wanneer status == 'needs_charging'.
        """
        target, _ = self._nearest_charging_node(self.current_node, heuristics)
        if target is None:
            raise Exception(
                f"[GSE {self.id}] Geen bereikbaar oplaadstation vanuit node {self.current_node}"
            )
        self.plan_to_node(
            target,
            nodes_dict,
            heuristics,
            t,
            stage="to_depot",
            label=f"charging station (node {target})",
        )

    # -------------------------------------------------------------------------
    # Veiling (Auction)
    # -------------------------------------------------------------------------

    def calculate_bid(self, gate_node_id, heuristics, second_node_id=None):
        """
        Berekent een bod voor een taak bij gate_node_id.
        Lagere waarde = beter bod.
        Biedt float('inf') als GSE niet beschikbaar, accu te laag, of als de GSE
        niet genoeg accu heeft om de taak te voltooien én terug naar het depot te rijden.
        INPUT:
            - gate_node_id  : [int] eerste doelnode (cargo_from bij load, plane.node_id bij unload)
            - heuristics    : [dict] voorberekende afstanden tussen nodes
            - second_node_id: [int] tweede doelnode (plane.node_id bij load, cargo_to bij unload)
        RETURNS:
            - bid : [float] bod (lagere waarde = beter)
        """
        if self.status != "available" or self.soc <= self.low_soc_threshold:
            return float('inf')

        # Controleer of de route bestaat
        if gate_node_id not in heuristics.get(self.current_node, {}):
            return float('inf')

        distance = heuristics[self.current_node][gate_node_id]
        if second_node_id is not None:
            if second_node_id not in heuristics.get(gate_node_id, {}):
                return float('inf')
            distance += heuristics[gate_node_id][second_node_id]
            last_node = second_node_id
        else:
            last_node = gate_node_id

        # Controleer of de GSE na de taak het dichtstbijzijnde oplaadstation kan bereiken
        _, return_distance = self._nearest_charging_node(last_node, heuristics)
        if return_distance == float('inf'):
            return float('inf')

        # Accu-verbruik per afstandseenheid is constant ongeacht snelheid
        energy_per_unit = self.base_consumption_rate  # = consumption_rate / speed
        energy_needed = (distance + return_distance) * energy_per_unit
        if energy_needed > self.soc:
            return float('inf')

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
            node_sequence = " -> ".join(str(int(node_id)) for node_id, _ in path)
            print(f"[GSE {self.id}] Pad naar {label}: {node_sequence}")
        else:
            raise Exception(f"[GSE {self.id}] Geen pad gevonden naar {label} (node {self.goal})")

        if path[0][1] != t:
            raise Exception(f"[GSE {self.id}] Tijdstip van pad klopt niet: verwacht {t}, kreeg {path[0][1]}")

    def plan_to_node(self, node_id, nodes_dict, heuristics, t, stage=None, label="goal"):
        self.goal = node_id
        # Als de GSE midden op een edge zit (positie ≠ huidige node), plan dan vanaf
        # het volgende knooppunt waarnaartoe al gereden wordt. Dit voorkomt dat het
        # nieuwe pad diagonaal loopt vanaf de tussenliggende positie.
        current_node_pos = self.nodes_dict[self.current_node]["xy_pos"]
        mid_edge = (math.dist(self.position, current_node_pos) > 1e-3
                    and self.from_to[0] != self.from_to[1])
        self.start = self.from_to[1] if mid_edge else self.current_node
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
        dx = xy_next[0] - xy_start[0]
        dy = xy_next[1] - xy_start[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return  # geen beweging, behoud huidige heading
        if abs(dx) < 1e-9:                      # puur verticaal
            self.heading = 0 if dy > 0 else 180
        elif abs(dy) < 1e-9:                    # puur horizontaal
            self.heading = 270 if dx > 0 else 90
        else:
            # Diagonale beweging (kan optreden bij herplanning midden op een edge).
            # Bereken de hoek met atan2 zodat we niet crashen.
            self.heading = round(math.degrees(math.atan2(-dx, dy)) % 360)

    def move(self, dt, t):
        """
        Beweegt de GSE één tijdstap langs het geplande pad.
        Werkt positie, heading en current_node bij.
        Bij aankomst op het doel wordt de status bijgewerkt.
        INPUT:
            - dt : grootte van de tijdstap
            - t  : huidig tijdstip (voor logging)
        """
        if not self.path_to_goal or dt <= 0:
            return

        remaining_travel = self.speed * dt
        while remaining_travel > 1e-9 and self.path_to_goal:
            from_node = self.from_to[0]
            to_node   = self.from_to[1]
            xy_to     = self.nodes_dict[to_node]["xy_pos"]
            distance_to_target = math.dist(self.position, xy_to)

            if distance_to_target <= 1e-9:
                self.position = xy_to
                self.current_node = to_node
                if to_node == self.goal:
                    self._on_goal_reached(t)
                    return
                self._advance_path_segment()
                continue

            self.get_heading(self.position, xy_to)
            step_len = min(remaining_travel, distance_to_target)
            ratio    = step_len / distance_to_target
            posx     = round(self.position[0] + (xy_to[0] - self.position[0]) * ratio, 2)
            posy     = round(self.position[1] + (xy_to[1] - self.position[1]) * ratio, 2)
            self.position = (posx, posy)
            remaining_travel -= step_len

            if step_len >= distance_to_target - 1e-9 or math.dist(self.position, xy_to) <= 1e-3:
                self.position = xy_to
                self.current_node = to_node

                if to_node == self.goal:
                    self._on_goal_reached(t)
                    return

                self._advance_path_segment()
            else:
                break

    def _advance_path_segment(self):
        """
        Schuif door naar het volgende edge-segment van het huidige pad.
        """
        if len(self.path_to_goal) <= 1:
            self.path_to_goal = []
            self.from_to = [self.current_node, self.current_node]
            return

        self.path_to_goal = self.path_to_goal[1:]
        new_next_id       = self.path_to_goal[0][0]
        self.last_node    = self.from_to[0]
        self.from_to      = [self.current_node, new_next_id]

    def _on_goal_reached(self, t):
        """
        Callback bij aankomst op het doelknooppunt.
        Bepaalt de nieuwe status op basis van het doel.
        """
        self.path_to_goal = []
        self.from_to = [self.current_node, self.current_node]

        if self.task_stage == "to_depot":
            # Veiligheidscheck: zorg dat we daadwerkelijk op een oplaadstation staan
            node_type = self.nodes_dict[self.current_node]["type"]
            if node_type != "charging":
                raise Exception(
                    f"[GSE {self.id}] t={t}: aankomst bij node {self.current_node} (type='{node_type}') "
                    f"maar dit is geen oplaadstation!"
                )
            self.status = "charging"
            self.charge_time_elapsed = 0.0
            self.work_end_time = None
            self.assigned_plane_id = None
            self.assigned_service_type = None
            self.task_stage = None
            print(f"[GSE {self.id}] t={t}: oplaadstation {self.current_node} bereikt, status -> charging "
                  f"(SoC={self.soc:.1f}%, duurt {self.charge_duration:.0f} min)")
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

        self.completed_tasks += 1

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
