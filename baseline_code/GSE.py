from single_agent_planner import simple_single_agent_astar
import math

class GSE(object):
    """
    GSE (Ground Support Equipment) agent used in the simulation.
    Combines path-planning and movement logic with battery (SoC) management
    and auction bidding for task assignment.
    """
    def __init__(self, gse_id, start_node, nodes_dict, charging_nodes=None, speed=1.0):
        # Fixed parameters
        self.id             = gse_id
        self.nodes_dict     = nodes_dict
        self.charging_nodes = list(charging_nodes) if charging_nodes else []
        self.base_speed  = 1.0          # reference speed used for battery scaling
        self.base_consumption_rate = 0.5  # consumption per time unit at base_speed
        self.speed       = self.base_speed

        # Start position
        self.start         = start_node
        self.current_node  = start_node
        self.position      = nodes_dict[start_node]["xy_pos"]

        # Status and task
        # Possible statuses:
        #   "available"      - ready for a new task
        #   "taxiing"        - driving to a gate or depot
        #   "working"        - servicing a plane at the gate
        #   "charging"       - charging at the depot
        #   "needs_charging" - battery low, must go to depot
        #   "at_cargo_pickup" / "at_cargo_dropoff" - intermediate cargo states
        self.status               = "available"
        self.goal                 = None    # current target node
        self.assigned_plane_id    = None
        self.assigned_service_type = None
        self.work_end_time        = None
        self.task_stage           = None

        # Battery
        self.soc              = 100.0   # State of Charge in %
        self.consumption_rate = self.base_consumption_rate
        self.charge_duration  = 15.0   # fixed charging duration in minutes (always full charge)
        self.charge_time_elapsed = 0.0  # elapsed charging time in current session
        self.low_soc_threshold   = 20.0  # below this, stop bidding and go to depot
        self.critical_soc_threshold = 10.0  # emergency: go to depot immediately while driving
        self.total_energy_consumed = 0.0  # cumulative battery consumption over the full simulation (%)
        self.completed_tasks       = 0    # number of completed service tasks

        # Path and movement
        self.path_to_goal = []   # list of (node_id, timestep) tuples
        self.from_to      = [0, 0]
        self.heading      = 0
        self.waiting      = False

        self.set_speed(speed)

    def set_speed(self, speed):
        # Speed and scale battery use so consumption per distance is constant
        self.speed = float(speed)
        speed_ratio = self.speed / self.base_speed
        self.consumption_rate = self.base_consumption_rate * speed_ratio

    # Battery management
    def update_soc(self, dt):
        # Update SoC based on current status, caled once per timestep
        if self.status == "taxiing":
            # Battery drains only while moving
            drain = min(self.soc, self.consumption_rate * dt)
            self.soc -= drain
            self.total_energy_consumed += drain

        elif self.status == "charging":
            # Fixed charging time of charge_duration minutes, then fully charged
            self.charge_time_elapsed += dt
            if self.charge_time_elapsed >= self.charge_duration:
                self.soc = 100.0
                self.charge_time_elapsed = 0.0
                self.status = "available"
                self.goal = None
                print(f"[GSE {self.id}] Volledig opgeladen na {self.charge_duration:.0f} minuten, status -> available")

        # Low battery (10-20%): send to depot when available or done working
        if self.soc <= self.low_soc_threshold and self.status == "available":
            self.status = "needs_charging"
            print(f"[GSE {self.id}] Lage SoC ({self.soc:.1f}%), status -> needs_charging")

        # Emergency: critically low battery (<10%) while driving to a task
        elif (self.soc <= self.critical_soc_threshold
              and self.status == "taxiing"
              and self.task_stage != "to_depot"):
            self.status = "needs_charging"
            print(f"[GSE {self.id}] Kritieke SoC ({self.soc:.1f}%) tijdens rijden, noodstop -> needs_charging")

    def _nearest_charging_node(self, from_node, heuristics, busy_nodes=None):
        # Return the least busy charging station
        best_id, best_score = None, float('inf')
        for cn in self.charging_nodes:
            dist = heuristics.get(from_node, {}).get(cn, float('inf'))
            if dist == float('inf'):
                continue
            load = (busy_nodes or {}).get(cn, 0)
            score = dist + load * 50
            if score < best_score:
                best_score = score
                best_id = cn
        best_dist = heuristics.get(from_node, {}).get(best_id, float('inf')) if best_id else float('inf')
        return best_id, best_dist

    def go_charge(self, nodes_dict, heuristics, t, busy_charging_nodes=None):
        # Plan a route to the least busy chargings station
        target, _ = self._nearest_charging_node(self.current_node, heuristics, busy_charging_nodes)
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

    # AUCTION
    def calculate_bid(
        self,
        gate_node_id,
        heuristics,
        second_node_id=None,
        alpha=1.0,
        beta=1.0,
        max_shortest_path_distance=1.0,
    ):
        # Lower bid = better. Returns infinity if unavailable, SoC too low, or battery insufficient for the route.
        # gate_node_id: first target (cargo_from for load, plane.node_id for unload)
        # second_node_id: second target (plane.node_id for load, cargo_to for unload)
        if self.status != "available" or self.soc <= self.low_soc_threshold:
            return float('inf')

        # Check whether the route exists
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

        # Energy per distance unit is constant regardless of speed.
        # Only the task route is priced here: finish_working sends the GSE to depot
        # afterward when SoC is low. Including return distance here would block
        # mid-SoC GSEs from taking otherwise feasible tasks.
        energy_per_unit = self.base_consumption_rate  # = consumption_rate / speed
        energy_needed = distance * energy_per_unit
        if energy_needed > self.soc:
            return float('inf')

        # Normalize both components and combine them with configurable weights.
        # distance in [0, 1+] via fixed d_max scale; SoC penalty in [0, 1].
        distance_norm = distance / max(max_shortest_path_distance, 1e-9)
        soc_penalty_norm = (100.0 - self.soc) / 100.0

        bid = alpha * distance_norm + beta * soc_penalty_norm
        return bid

    # Path planning
    def _plan_path(self, nodes_dict, heuristics, t, label="goal", forbidden_nodes=None):
        # Runs A* from self.start to self.goal; forbidden_nodes optionally excludes nodes (used for rerouting)
        success, path = simple_single_agent_astar(
            nodes_dict, self.start, self.goal, heuristics, t, forbidden_nodes=forbidden_nodes, speed=self.speed
        )
        if success:
            mid_edge = (self.start != self.current_node)
            if mid_edge:
                # GSE is mid-edge between current_node and from_to[1] (= self.start).
                # Keep from_to unchanged so the GSE finishes the current edge first.
                # Store the full path (including self.start) so _advance_path_segment
                # correctly pops each hop when the GSE arrives at each node.
                self.path_to_goal = path
            else:
                self.path_to_goal = path[1:]
                if not self.path_to_goal:
                    self.from_to = [path[0][0], path[0][0]]
                    self.current_node = path[0][0]
                    self.position = self.nodes_dict[self.current_node]["xy_pos"]
                    self._on_goal_reached(t)
                    return
                next_node_id  = self.path_to_goal[0][0]
                self.from_to  = [path[0][0], next_node_id]
            node_sequence = " -> ".join(str(int(node_id)) for node_id, _ in path)
            print(f"[GSE {self.id}] Path to {label}: {node_sequence}")
        else:
            raise Exception(f"[GSE {self.id}] No path found to {label} (node {self.goal})")

        if path[0][1] != t:
            raise Exception(f"[GSE {self.id}] Path timestamp mismatch: expected {t}, got {path[0][1]}")

    def plan_to_node(self, node_id, nodes_dict, heuristics, t, stage=None, label="goal", forbidden_nodes=None):
        self.goal = node_id
        # If the GSE is mid-edge (position != current node), replan from
        # the next node already being approached. This avoids diagonal jumps
        # from an in-between position.
        current_node_pos = self.nodes_dict[self.current_node]["xy_pos"]
        mid_edge = (math.dist(self.position, current_node_pos) > 1e-3
                    and self.from_to[0] != self.from_to[1])
        self.start = self.from_to[1] if mid_edge else self.current_node
        self.status = "taxiing"
        self.task_stage = stage
        self.work_end_time = None
        self._plan_path(nodes_dict, heuristics, t, label=label, forbidden_nodes=forbidden_nodes)

    # Plans the GSE's route based on service type:
    #   load: drive to cargo_from first (pick up cargo), then continue to the plane
    #   unload: drive directly to the plane (cargo is already on board)
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

    # Movement
    # Computes heading in degrees from xy_start to xy_next and updates self.heading
    def get_heading(self, xy_start, xy_next):
        dx = xy_next[0] - xy_start[0]
        dy = xy_next[1] - xy_start[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return  # no movement, keep current heading
        if abs(dx) < 1e-9:                      # purely vertical
            self.heading = 0 if dy > 0 else 180
        elif abs(dy) < 1e-9:                    # purely horizontal
            self.heading = 270 if dx > 0 else 90
        else:
            # Diagonal movement can occur during mid-edge replanning.
            # Use atan2 for a stable heading calculation.
            self.heading = round(math.degrees(math.atan2(-dx, dy)) % 360)

    # Advances the GSE along its path by up to speed*dt distance units.
    # May cross multiple path segments in one timestep if dt is large.
    def move(self, dt, t):
        if not self.path_to_goal or dt <= 0:
            return

        remaining_travel = self.speed * dt
        while remaining_travel > 1e-9 and self.path_to_goal:
            to_node   = self.from_to[1]
            xy_to     = self.nodes_dict[to_node]["xy_pos"]
            distance_to_target = math.dist(self.position, xy_to)

            # Already at the next node — snap and advance without consuming travel budget
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
        # Advance to the next edge segment of the current path
        if len(self.path_to_goal) <= 1:
            self.path_to_goal = []
            self.from_to = [self.current_node, self.current_node]
            return

        self.path_to_goal = self.path_to_goal[1:]
        new_next_id       = self.path_to_goal[0][0]
        self.from_to      = [self.current_node, new_next_id]

    def _on_goal_reached(self, t):
        # Callback when arriving at the goal node.
        # Sets the next status based on task stage.
        self.path_to_goal = []
        self.from_to = [self.current_node, self.current_node]

        if self.task_stage == "to_depot":
            self.status = "charging"
            self.charge_time_elapsed = 0.0
            self.work_end_time = None
            self.assigned_plane_id = None
            self.assigned_service_type = None
            self.task_stage = None
            print(f"[GSE {self.id}] t={t}: charging station {self.current_node} reached, status -> charging "
                f"(SoC={self.soc:.1f}%, duration {self.charge_duration:.0f} min)")
        elif self.task_stage == "load_to_cargo":
            self.status = "at_cargo_pickup"
            print(f"[GSE {self.id}] t={t}: cargo pickup reached for plane {self.assigned_plane_id}")
        elif self.task_stage == "unload_to_cargo":
            self.status = "at_cargo_dropoff"
            print(f"[GSE {self.id}] t={t}: cargo dropoff reached for plane {self.assigned_plane_id}")
        else:
            # Gate reached: start servicing the plane
            self.status = "working"
            service_label = self.assigned_service_type or "service"
            print(f"[GSE {self.id}] t={t}: gate {self.goal} bereikt, status -> working ({service_label})")


    def finish_working(self, nodes_dict, heuristics, t):
        # When the assigned plane service has finished, the GSE returns to the depot or becomes available
        self.assigned_plane_id = None
        self.assigned_service_type = None
        self.work_end_time = None
        self.task_stage = None

        self.completed_tasks += 1

        if self.soc < self.low_soc_threshold:
            # Battery too low: return to depot immediately
            self.go_charge(nodes_dict, heuristics, t)
        else:
            self.status = "available"
            self.goal   = None
            print(f"[GSE {self.id}] t={t}: task completed, status -> available (SoC={self.soc:.1f}%)")

    def __repr__(self):
        return (f"GSE(id={self.id}, status={self.status}, "
                f"node={self.current_node}, SoC={self.soc:.1f}%)")
