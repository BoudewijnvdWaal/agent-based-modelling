from pathlib import Path

import pandas as pd


class Plane:
    """Static aircraft parked at a gate, spawned from Plane_data.xlsx."""

    def __init__(
        self,
        plane_id,
        gate_node_id,
        spawn_time_minutes,
        cargo_from,
        cargo_to,
        nodes_dict,
        unloading_duration_minutes,
        loading_duration_minutes,
        departure_delay_minutes,
    ):
        gate_node_id = int(gate_node_id)
        self.id = str(plane_id)
        self.node_id = gate_node_id
        self.gate_node_id = gate_node_id
        self.spawn_time = float(spawn_time_minutes)
        self.spawn_time_minutes = float(spawn_time_minutes)
        self.cargo_from = int(cargo_from)
        self.cargo_to = int(cargo_to)

        self.xy_pos = nodes_dict[gate_node_id]["xy_pos"]
        self.unloading_duration_minutes = float(unloading_duration_minutes)
        self.loading_duration_minutes = float(loading_duration_minutes)
        self.departure_delay_minutes = float(departure_delay_minutes)
        self.current_service_type = None
        self.current_service_end_time = None
        self.current_service_gse_id = None
        self.assigned_gse_id = None
        self.unload_start_time = None
        self.load_start_time = None
        self.unloading_complete_time = None
        self.loading_complete_time = None
        self.departure_ready_time = None
        self.departed_time = None
        self.load_release_gse_id = None
        self.status = "scheduled"

    # Called when the plane arrives at its gate; begins the unload phase
    def spawn(self):
        self.status = "awaiting_unload"
        return self

    # Returns the next required service type based on current status
    @property
    def next_service_type(self):
        # Planes always follow unload -> load.
        if self.status == "awaiting_unload":
            return "unload"
        if self.status == "awaiting_load":
            return "load"
        return None

    # Returns True if the plane still needs a GSE assigned
    def needs_service_assignment(self):
        return self.next_service_type is not None

    # Locks in a GSE assignment and advances the plane's status
    def mark_service_assigned(self, service_type, gse_id):
        if service_type == "unload":
            self.status = "unload_assigned"
        elif service_type == "load":
            self.status = "load_assigned"
        self.assigned_gse_id = gse_id

    # Starts the service, records timing, and returns the computed end time
    def start_service(self, service_type, gse_id, start_time):
        if service_type == "unload":
            active_status = "unloading"
            duration = self.unloading_duration_minutes
        elif service_type == "load":
            active_status = "loading"
            duration = self.loading_duration_minutes
        self.status = active_status
        self.current_service_type = service_type
        self.current_service_gse_id = gse_id
        self.current_service_end_time = float(start_time) + duration
        if service_type == "unload":
            self.unload_start_time = float(start_time)
        elif service_type == "load":
            self.load_start_time = float(start_time)
        return self.current_service_end_time

    # Finalises the current service; after unload, waits for GSE to leave before allowing load
    def complete_current_service(self, completion_time):
        if self.current_service_type == "unload":
            self.unloading_complete_time = float(completion_time)
            # Loading starts only after the unloading GSE leaves the gate
            self.status = "awaiting_load_release"
        elif self.current_service_type == "load":
            self.loading_complete_time = float(completion_time)
            self.departure_ready_time = self.loading_complete_time + self.departure_delay_minutes
            self.status = "ready_to_depart"
        completed_service_type = self.current_service_type
        self.current_service_type = None
        self.current_service_end_time = None
        self.current_service_gse_id = None
        self.assigned_gse_id = None
        return completed_service_type

    # Records which GSE finished unloading so the gate can be cleared before loading begins
    def mark_unloading_departed(self, gse_id):
        self.load_release_gse_id = gse_id

    # Clears the unload GSE and opens the plane for load assignment
    def release_for_loading(self):
        self.status = "awaiting_load"
        self.load_release_gse_id = None

    # Returns True when the plane is ready to leave and departure time has passed
    def ready_to_despawn(self, t):
        return (
            self.status == "ready_to_depart"
            and self.departure_ready_time is not None
            and t >= self.departure_ready_time - 1e-9
        )

    def mark_departed(self, departure_time):
        self.departed_time = float(departure_time)
        self.status = "departed"

    # Total time from gate arrival to departure in simulation minutes
    @property
    def turnaround_time(self):
        if self.departed_time is None:
            return None
        return self.departed_time - self.spawn_time

    def to_gate_state(self):
        return {
            "id": self.id,
            "node_id": self.node_id,
            "xy_pos": self.xy_pos,
        }

    def __repr__(self):
        return (
            f"Plane(id={self.id}, gate={self.node_id}, spawn={self.spawn_time_minutes}, "
            f"cargo_from={self.cargo_from}, cargo_to={self.cargo_to}, status={self.status})"
        )


def _resolve_path(path_like, base_dir=None):
    path = Path(path_like).expanduser()
    candidates = [path]
    if not path.is_absolute() and base_dir is not None:
        candidates.append(Path(base_dir) / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find plane data file '{path_like}'. "
        f"Tried: {', '.join(str(candidate) for candidate in candidates)}"
    )


def _normalize_columns(columns):
    return {
        column: column.strip().lower().replace(" ", "_")
        for column in columns
    }


def load_plane_schedule(
    plane_data_file,
    nodes_dict,
    unloading_duration_minutes,
    loading_duration_minutes,
    departure_delay_minutes,
    base_dir=None,
):
    plane_data_path = _resolve_path(plane_data_file, base_dir=base_dir)
    df_planes = pd.read_excel(plane_data_path)
    df_planes = df_planes.rename(columns=_normalize_columns(df_planes.columns))

    required_columns = {"plane", "gate", "sibt", "cargo_from", "cargo_to"}
    missing_columns = required_columns - set(df_planes.columns)
    if missing_columns:
        raise ValueError(
            "Plane_data file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    plane_schedule = []
    sorted_planes = df_planes.sort_values(["sibt", "plane"], kind="stable")
    for _, row in sorted_planes.iterrows():
        plane_schedule.append(
            Plane(
                plane_id=row["plane"],
                gate_node_id=row["gate"],
                spawn_time_minutes=row["sibt"],
                cargo_from=row["cargo_from"],
                cargo_to=row["cargo_to"],
                nodes_dict=nodes_dict,
                unloading_duration_minutes=unloading_duration_minutes,
                loading_duration_minutes=loading_duration_minutes,
                departure_delay_minutes=departure_delay_minutes,
            )
        )

    return plane_schedule


def build_plane_schedule_lookup(plane_schedule):
    schedule_lookup = {}
    for plane in plane_schedule:
        # Key on 2 decimals so this matches the t-values in run_me exactly.
        schedule_lookup.setdefault(round(plane.spawn_time, 2), []).append(plane)
    return schedule_lookup


def spawn_planes(t, schedule_lookup):
    spawned_planes = schedule_lookup.pop(t, [])
    for plane in spawned_planes:
        plane.spawn()
    return spawned_planes
