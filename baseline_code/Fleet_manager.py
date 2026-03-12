class Fleet_manager:
    def __init__(self, nodes_dict, planner=None):
        self.planner = planner
        # Collect gate node ids from the layout
        self.gate_ids = sorted([nid for nid, props in nodes_dict.items() if props.get("type") == "gate"])
        # Binary occupancy: 1 = occupied, 0 = free
        self.gate_status = {gid: 0 for gid in self.gate_ids}

    def update_gate_status(self, gate_planes, aircraft_lst=None, t=None):
        """
        Update the binary occupancy map.
        gate_planes: list of dicts with key 'node_id' for parked planes.
        aircraft_lst: optional list of GSE instances; gates where a/c have status 'arrived' are marked occupied too.
        t: optional current simulation time (for logging)
        """
        prev_status = self.gate_status
        status = {gid: 0 for gid in self.gate_ids}  # start with all free
        # mark static gate planes
        for gp in gate_planes:
            gid = gp.get("node_id")
            if gid in status:
                status[gid] = 1
                if prev_status.get(gid, 0) == 0:
                    status_list = [status[g] for g in self.gate_ids]
                    print(f"[FleetManager] t={t}: gate {gid} occupied by parked plane id={gp.get('id')} | status={status_list}")
        # mark arrived aircraft parked at a gate
        if aircraft_lst:
            for ac in aircraft_lst:
                if getattr(ac, "status", None) == "arrived" and ac.goal in status:
                    status[ac.goal] = 1
                    if prev_status.get(ac.goal, 0) == 0:
                        status_list = [status[g] for g in self.gate_ids]
                        print(f"[FleetManager] t={t}: gate {ac.goal} occupied by aircraft id={ac.id} | status={status_list}")
        self.gate_status = status

    def gate_occupied(self, gate_id):
        """Return True if the given gate_id is currently occupied."""
        return self.gate_status.get(gate_id, 0) == 1

    def gate_status_list(self):
        """Return occupancy as a list ordered by sorted gate ids."""
        return [self.gate_status[gid] for gid in self.gate_ids]
