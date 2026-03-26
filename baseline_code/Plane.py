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
        turnaround_time,
    ):
        gate_node_id = int(gate_node_id)
        if gate_node_id not in nodes_dict:
            raise ValueError(
                f"Gate node {gate_node_id} for plane {plane_id} not found in nodes_dict."
            )

        self.id = str(plane_id)
        self.node_id = gate_node_id
        self.gate_node_id = gate_node_id
        self.spawn_time = float(spawn_time_minutes)
        self.spawn_time_minutes = float(spawn_time_minutes)
        self.cargo_from = int(cargo_from)
        self.cargo_to = int(cargo_to)
        self.xy_pos = nodes_dict[gate_node_id]["xy_pos"]
        self.turnaround_time = float(turnaround_time)
        self.despawn_time = self.spawn_time + self.turnaround_time
        self.serviced = False
        self.status = "scheduled"

    def spawn(self):
        self.status = "parked"
        return self

    def ready_to_despawn(self, t, working_gate_nodes):
        return (
            t >= self.despawn_time - 1e-9
            and (not self.serviced or self.node_id in working_gate_nodes)
        )

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


def load_plane_schedule(plane_data_file, nodes_dict, turnaround_time, base_dir=None):
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
                turnaround_time=turnaround_time,
            )
        )

    return plane_schedule


def build_plane_schedule_lookup(plane_schedule):
    schedule_lookup = {}
    for plane in plane_schedule:
        schedule_lookup.setdefault(round(plane.spawn_time, 2), []).append(plane)
    return schedule_lookup


def spawn_planes(t, schedule_lookup):
    spawned_planes = schedule_lookup.pop(t, [])
    for plane in spawned_planes:
        plane.spawn()
    return spawned_planes
