"""
batch_run.py — Paired batch runner: electric vs. conventional GSE.

Usage:
    python batch_run.py [--seeds N] [--output results.csv]

Defaults: 30 seeds, output to batch_results.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOG_DIR  = PROJECT_ROOT / "run logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LATEST_BATCH_FILE = LOG_DIR / "latest_batch.txt"
SUMMARY_LOG_NAME = "batch_summary.log"
PARAMETERS_CSV = "parameters.csv"
PLANES_SUMMARY_CSV = "planes_summary.csv"
GSE_SUMMARY_CSV = "GSE_summary.csv"

# =============================================================================
# BATCH PARAMETERS  (change here)
# =============================================================================
N_SEEDS    = 115                  # number of paired runs (seeds 0 … N_SEEDS-1)
N_WORKERS  = max(1, os.cpu_count() - 1)  # parallel workers; -1 leaves one core for the OS
OUTPUT_CSV = "batch_results.csv" # filename inside the batch folder
# =============================================================================


# =============================================================================
# LOG PARSER  (2.2)
# =============================================================================

def parse_log(log_path: Path) -> dict:
    """
    Parse a run log file. Returns:
      gse_type    : "electric" or "gas"
      mean_tat    : float | None
      total_tat   : float | None
      n_completed : int
      flights     : list of {id, tat, wait_unload, wait_load}
      gse_energy  : list of {id, energy_pct, tasks}
    """
    text = log_path.read_text()

    result = {
        "gse_type":    None,
        "date_time":   None,
        "duration_hours": None,
        "duration_min": None,
        "gse_count":   None,
        "gse_speed":   None,
        "charge_duration_min": None,
        "base_consumption_rate_pct_per_unit": None,
        "recharge_rate_pct_per_min": None,
        "mean_tat":    None,
        "total_tat":   None,
        "n_completed": 0,
        "flights":     [],
        "gse_energy":  [],
    }

    m = re.search(r"GSE type\s*:\s*(\S+)", text)
    if m:
        result["gse_type"] = m.group(1).lower()

    m = re.search(r"Date/time\s*:\s*(.+)", text)
    if m:
        result["date_time"] = m.group(1).strip()

    m = re.search(r"Duration\s*:\s*([\d.]+)\s+hours\s+\(([\d.]+)\s+min\)", text)
    if m:
        result["duration_hours"] = float(m.group(1))
        result["duration_min"] = float(m.group(2))

    m = re.search(r"GSE count\s*:\s*(\d+)\s+\|\s+speed:\s*([\d.]+)", text)
    if m:
        result["gse_count"] = int(m.group(1))
        result["gse_speed"] = float(m.group(2))

    m = re.search(
        r"GSE params\s*:\s*charge_duration=([\d.]+)\s+min\s+\|\s+base_consumption=([\d.]+)%/unit",
        text,
    )
    if m:
        result["charge_duration_min"] = float(m.group(1))
        result["base_consumption_rate_pct_per_unit"] = float(m.group(2))
        if result["charge_duration_min"]:
            result["recharge_rate_pct_per_min"] = 100.0 / result["charge_duration_min"]

    flight_re = re.compile(
        r"^\s{2}(\S+)\s+"
        r"(?:(\d+\.\d+)\s+min|incomplete)\s+"
        r"(?:(\d+\.\d+)\s+min|incomplete)\s+"
        r"(?:(\d+\.\d+)\s+min|incomplete)",
        re.MULTILINE,
    )
    for m in flight_re.finditer(text):
        fid = m.group(1)
        if fid in ("Flight", "Total", "Average"):
            continue
        result["flights"].append({
            "id":          fid,
            "tat":         float(m.group(2)) if m.group(2) else None,
            "wait_unload": float(m.group(3)) if m.group(3) else None,
            "wait_load":   float(m.group(4)) if m.group(4) else None,
        })

    m = re.search(r"Total \((\d+) completed\)\s+([\d.]+)\s+min", text)
    if m:
        result["n_completed"] = int(m.group(1))
        result["total_tat"]   = float(m.group(2))

    m = re.search(r"Average\s+([\d.]+)\s+min", text)
    if m:
        result["mean_tat"] = float(m.group(1))

    for m in re.finditer(r"GSE\s+(\d+)\s+([\d.]+)%\s+\[.*?\]\s+tasks:\s+(\d+)", text):
        result["gse_energy"].append({
            "id":     int(m.group(1)),
            "energy": float(m.group(2)),
            "tasks":  int(m.group(3)),
        })

    return result


# =============================================================================
# SILENCE HELPER  — suppresses run_me's terminal output during batch runs
# =============================================================================

@contextmanager
def _silence():
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()


# =============================================================================
# WORKER  — runs in a separate process, handles both conditions for one seed
# =============================================================================

def _run_seed(seed: int, batch_dir_str: str) -> dict:
    batch_dir = Path(batch_dir_str)

    with _silence():
        import run_me  # imported fresh per worker process

    log_conv = batch_dir / f"seed{seed:03d}_gas.log"
    log_elec = batch_dir / f"seed{seed:03d}_electric.log"

    spawn_cfg = run_me.build_random_gse_spawn_config(
        run_me.nodes_dict, run_me.GSE_COUNT, seed=seed
    )
    with _silence():
        run_me.run_simulation(
            2.0,
            0.25,
            "gas",
            log_filepath=log_conv,
            gse_spawn_config=spawn_cfg,
            enable_visualization=False,
            pace=False,
            show_status=False,
        )
    with _silence():
        run_me.run_simulation(
            15.0,
            0.5,
            "electric",
            log_filepath=log_elec,
            gse_spawn_config=spawn_cfg,
            enable_visualization=False,
            pace=False,
            show_status=False,
        )

    conv = parse_log(log_conv)
    elec = parse_log(log_elec)

    return {
        "seed":             seed,
        "tat_conventional": conv["mean_tat"],
        "tat_electric":     elec["mean_tat"],
        "n_completed_conv": conv["n_completed"],
        "n_completed_elec": elec["n_completed"],
        "log_conv":         log_conv.name,
        "log_elec":         log_elec.name,
        "parameter_rows":   [
            _parameters_row(seed, log_conv.name, conv),
            _parameters_row(seed, log_elec.name, elec),
        ],
        "plane_rows":       _plane_rows(seed, log_conv.name, conv) + _plane_rows(seed, log_elec.name, elec),
        "gse_rows":         _gse_rows(seed, log_conv.name, conv) + _gse_rows(seed, log_elec.name, elec),
    }


# =============================================================================
# PROGRESS BAR
# =============================================================================

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = round(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def _append_csv_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _parameters_row(seed: int, log_name: str, parsed: dict) -> dict:
    return {
        "seed": seed,
        "gse_type": parsed["gse_type"],
        "log_file": log_name,
        "date_time": parsed["date_time"],
        "duration_hours": parsed["duration_hours"],
        "duration_min": parsed["duration_min"],
        "gse_count": parsed["gse_count"],
        "gse_speed": parsed["gse_speed"],
        "charge_duration_min": parsed["charge_duration_min"],
        "base_consumption_rate_pct_per_unit": parsed["base_consumption_rate_pct_per_unit"],
        "recharge_rate_pct_per_min": parsed["recharge_rate_pct_per_min"],
    }


def _plane_rows(seed: int, log_name: str, parsed: dict) -> list[dict]:
    rows = []
    for flight in parsed["flights"]:
        rows.append({
            "seed": seed,
            "gse_type": parsed["gse_type"],
            "log_file": log_name,
            "plane_id": flight["id"],
            "completed": flight["tat"] is not None,
            "turnaround_time_min": flight["tat"],
            "wait_unload_min": flight["wait_unload"],
            "wait_load_min": flight["wait_load"],
        })
    return rows


def _gse_rows(seed: int, log_name: str, parsed: dict) -> list[dict]:
    rows = []
    for gse in parsed["gse_energy"]:
        rows.append({
            "seed": seed,
            "gse_type": parsed["gse_type"],
            "log_file": log_name,
            "gse_id": gse["id"],
            "total_energy_consumed_pct": gse["energy"],
            "tasks_completed": gse["tasks"],
        })
    return rows


def _write_batch_summary_header(summary_log: Path, batch_dir: Path, n_seeds: int,
                                effective_workers: int, output_csv: Path, batch_ts: str) -> None:
    summary_log.write_text(
        "\n".join([
            "=" * 60,
            "BATCH RUN SUMMARY",
            f"Created     : {batch_ts}",
            f"Seeds       : {n_seeds}",
            f"Workers     : {effective_workers}",
            f"Batch dir   : {batch_dir}",
            f"Results CSV : {output_csv}",
            "-" * 60,
            "seed,gas_mean_tat,electric_mean_tat,n_completed_gas,n_completed_electric,gas_log,electric_log",
        ]) + "\n"
    )


def _append_batch_summary_row(summary_log: Path, row: dict) -> None:
    with open(summary_log, "a") as f:
        f.write(
            f"{row['seed']},"
            f"{row['tat_conventional']},"
            f"{row['tat_electric']},"
            f"{row['n_completed_conv']},"
            f"{row['n_completed_elec']},"
            f"{row['log_conv']},"
            f"{row['log_elec']}\n"
        )


# =============================================================================
# BATCH RUNNER  (2.1)
# =============================================================================

def batch_run(n_seeds: int, n_workers: int, output_csv: Path) -> None:
    batch_start = time.monotonic()
    rows        = []
    effective_workers = max(1, min(n_workers, n_seeds))

    batch_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = LOG_DIR / f"batch_{batch_ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    output_csv = batch_dir / output_csv.name
    summary_log = batch_dir / SUMMARY_LOG_NAME
    parameters_csv = batch_dir / PARAMETERS_CSV
    planes_summary_csv = batch_dir / PLANES_SUMMARY_CSV
    gse_summary_csv = batch_dir / GSE_SUMMARY_CSV
    _write_batch_summary_header(summary_log, batch_dir, n_seeds, effective_workers, output_csv, batch_ts)
    LATEST_BATCH_FILE.write_text(
        f"{batch_dir}\n"
        f"results_csv={output_csv}\n"
        f"summary_log={summary_log}\n"
        f"parameters_csv={parameters_csv}\n"
        f"planes_summary_csv={planes_summary_csv}\n"
        f"gse_summary_csv={gse_summary_csv}\n"
        f"created_at={batch_ts}\n"
        f"seeds={n_seeds}\n"
        f"workers={effective_workers}\n"
    )

    print(f"Batch run: {n_seeds} seeds × 2 conditions  |  {effective_workers} parallel workers")
    print(f"Output folder: {batch_dir}\n")

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_run_seed, seed, str(batch_dir)): seed
            for seed in range(n_seeds)
        }

        for future in as_completed(futures):
            row = future.result()
            summary_row = {
                "seed": row["seed"],
                "tat_conventional": row["tat_conventional"],
                "tat_electric": row["tat_electric"],
                "n_completed_conv": row["n_completed_conv"],
                "n_completed_elec": row["n_completed_elec"],
                "log_conv": row["log_conv"],
                "log_elec": row["log_elec"],
            }
            seed = summary_row["seed"]
            rows.append(summary_row)

            done    = len(rows)
            elapsed = time.monotonic() - batch_start
            avg_per_seed = elapsed / done
            eta     = avg_per_seed * (n_seeds - done)

            print(
                f"  {_bar(done, n_seeds)} {done}/{n_seeds}  "
                f"seed {seed:>3}  gas={summary_row['tat_conventional']}  elec={summary_row['tat_electric']}  "
                f"elapsed: {_fmt_duration(elapsed)}  ETA: {_fmt_duration(eta)}"
            )

            rows_sorted = sorted(rows, key=lambda r: r["seed"])
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                writer.writeheader()
                writer.writerows(rows_sorted)
            _append_batch_summary_row(summary_log, summary_row)
            _append_csv_rows(parameters_csv, row["parameter_rows"])
            _append_csv_rows(planes_summary_csv, row["plane_rows"])
            _append_csv_rows(gse_summary_csv, row["gse_rows"])

    total = _fmt_duration(time.monotonic() - batch_start)
    print(f"\nDone in {total}. Results saved to {output_csv}")
    print(f"Batch summary: {summary_log}")
    print(f"Parameters CSV: {parameters_csv}")
    print(f"Planes summary CSV: {planes_summary_csv}")
    print(f"GSE summary CSV: {gse_summary_csv}")
    print(f"Latest batch pointer: {LATEST_BATCH_FILE}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   type=int, default=N_SEEDS,   help=f"Number of random seeds (default: {N_SEEDS})")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help=f"Parallel workers (default: {N_WORKERS})")
    parser.add_argument("--output",  type=str, default=OUTPUT_CSV, help=f"Output CSV (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    batch_run(n_seeds=args.seeds, n_workers=args.workers, output_csv=BASE_DIR / args.output)
