"""
experiment_1_analysis.py — Experiment 1: Paired t-test electric vs gas GSE fleet.

Reads the latest batch run (or a specified batch folder) produced by batch_run.py,
performs a two-tailed paired t-test on mean TAT, and saves a full report.

Usage:
    python experiment_1_analysis.py                     # use latest_batch.txt pointer
    python experiment_1_analysis.py --batch <path>      # specify batch folder
    python experiment_1_analysis.py --csv   <path>      # specify batch_results.csv directly
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOG_DIR      = PROJECT_ROOT / "run logs"
LATEST_BATCH = LOG_DIR / "latest_batch.txt"


# =============================================================================
# LOCATE DATA
# =============================================================================

def _resolve_csv(batch_arg: str | None, csv_arg: str | None) -> tuple[Path, Path]:
    """Return (batch_results_csv, batch_dir)."""
    if csv_arg:
        p = Path(csv_arg)
        return p, p.parent
    if batch_arg:
        d = Path(batch_arg)
        return d / "batch_results.csv", d
    if LATEST_BATCH.exists():
        lines = {k: v for k, v in (
            line.strip().split("=", 1)
            for line in LATEST_BATCH.read_text().splitlines()
            if "=" in line
        )}
        csv_path = Path(lines["results_csv"])
        return csv_path, csv_path.parent
    raise FileNotFoundError(
        "No batch data found. Run batch_run.py first, or pass --batch / --csv."
    )


# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1))


def mean_ci_95(values: np.ndarray) -> tuple[float, float]:
    n    = len(values)
    se   = values.std(ddof=1) / math.sqrt(n)
    t_cv = stats.t.ppf(0.975, df=n - 1)
    return float(values.mean() - t_cv * se), float(values.mean() + t_cv * se)


def diff_ci_95(diff: np.ndarray) -> tuple[float, float]:
    return mean_ci_95(diff)


# =============================================================================
# FORMATTING
# =============================================================================

SEP_THICK = "=" * 72
SEP_THIN  = "-" * 72

def _f(v, decimals=2) -> str:
    return f"{v:.{decimals}f}" if v is not None and not math.isnan(v) else "—"


def _pvalue_str(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.4f}"


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyse(csv_path: Path, batch_dir: Path) -> str:
    df = pd.read_csv(csv_path)

    # Drop any rows where TAT is missing (incomplete seeds)
    df_complete = df.dropna(subset=["tat_conventional", "tat_electric"]).copy()
    n_total   = len(df)
    n_valid   = len(df_complete)
    n_dropped = n_total - n_valid

    gas  = df_complete["tat_conventional"].to_numpy()
    elec = df_complete["tat_electric"].to_numpy()
    diff = gas - elec   # positive = gas slower

    # Paired t-test (two-tailed)
    t_stat, p_val = stats.ttest_rel(gas, elec)
    df_t          = n_valid - 1
    d             = cohens_d_paired(gas, elec)

    gas_ci   = mean_ci_95(gas)
    elec_ci  = mean_ci_95(elec)
    diff_ci  = diff_ci_95(diff)

    # Per-flight breakdown from planes_summary.csv (if present)
    planes_csv = batch_dir / "planes_summary.csv"
    flight_stats: dict = {}
    if planes_csv.exists():
        pf = pd.read_csv(planes_csv)
        for gse_type, grp in pf.groupby("gse_type"):
            completed = grp[grp["completed"] == True]
            flight_stats[gse_type] = {
                "n_flights":   len(completed),
                "mean_tat":    completed["turnaround_time_min"].mean(),
                "median_tat":  completed["turnaround_time_min"].median(),
                "std_tat":     completed["turnaround_time_min"].std(ddof=1),
                "min_tat":     completed["turnaround_time_min"].min(),
                "max_tat":     completed["turnaround_time_min"].max(),
                "mean_wait_unload": completed["wait_unload_min"].mean(),
                "mean_wait_load":   completed["wait_load_min"].mean(),
            }

    # GSE energy breakdown (if present)
    gse_csv = batch_dir / "GSE_summary.csv"
    gse_stats: dict = {}
    if gse_csv.exists():
        gf = pd.read_csv(gse_csv)
        for gse_type, grp in gf.groupby("gse_type"):
            gse_stats[gse_type] = {
                "mean_energy": grp["total_energy_consumed_pct"].mean(),
                "std_energy":  grp["total_energy_consumed_pct"].std(ddof=1),
                "mean_tasks":  grp["tasks_completed"].mean(),
                "std_tasks":   grp["tasks_completed"].std(ddof=1),
            }

    # =========================================================================
    # BUILD REPORT STRING
    # =========================================================================
    lines: list[str] = []
    a = lines.append

    a(SEP_THICK)
    a("EXPERIMENT 1 — PRIMARY HYPOTHESIS TEST")
    a("H0: Electric GSE fleet produces equivalent mean TAT to gas fleet")
    a(f"Analysis run : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"Batch folder : {batch_dir}")
    a(f"Data file    : {csv_path.name}")
    a(SEP_THICK)

    # ---- Sample ----
    a("")
    a("SAMPLE")
    a(SEP_THIN)
    a(f"  Total seeds run      : {n_total}")
    a(f"  Valid paired seeds   : {n_valid}")
    a(f"  Dropped (incomplete) : {n_dropped}")
    a(f"  GSE count            : 5  |  speed: 4.0")
    a(f"  Electric charge      : 15 min  |  Gas charge: 2 min")
    a(f"  Simulation duration  : 6 h (360 ticks)")

    # ---- Descriptive stats ----
    a("")
    a("DESCRIPTIVE STATISTICS  (mean TAT per seed, in minutes)")
    a(SEP_THIN)
    a(f"  {'Condition':<22} {'N':>5} {'Mean':>8} {'SD':>8} {'Median':>8} {'Min':>8} {'Max':>8}  95% CI")
    a(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'─'*20}")
    for label, arr, ci in [("Gas (conventional)", gas, gas_ci), ("Electric", elec, elec_ci)]:
        a(f"  {label:<22} {len(arr):>5} {_f(arr.mean()):>8} {_f(arr.std(ddof=1)):>8} "
          f"{_f(np.median(arr)):>8} {_f(arr.min()):>8} {_f(arr.max()):>8}  "
          f"[{_f(ci[0])}, {_f(ci[1])}]")
    a(f"")
    a(f"  {'Difference (gas − elec)':<22} {len(diff):>5} {_f(diff.mean()):>8} {_f(diff.std(ddof=1)):>8} "
      f"{_f(np.median(diff)):>8} {_f(diff.min()):>8} {_f(diff.max()):>8}  "
      f"[{_f(diff_ci[0])}, {_f(diff_ci[1])}]")

    # ---- Inferential stats ----
    a("")
    a("PAIRED t-TEST  (two-tailed, α = 0.05)")
    a(SEP_THIN)
    a(f"  t({df_t}) = {t_stat:+.4f}")
    a(f"  p-value  = {_pvalue_str(p_val)}")
    a(f"  Cohen's d (paired) = {d:+.4f}  [{_effect_label(d)} effect]")
    a(f"  95% CI of mean difference: [{_f(diff_ci[0])}, {_f(diff_ci[1])}] min")
    a("")
    if p_val < 0.05:
        direction = "GAS is slower" if diff.mean() > 0 else "ELECTRIC is slower"
        a(f"  RESULT: Statistically significant (p {_pvalue_str(p_val)} < 0.05).")
        a(f"          Reject H0. {direction} on average.")
    else:
        a(f"  RESULT: Not statistically significant (p {_pvalue_str(p_val)} >= 0.05).")
        a(f"          Fail to reject H0. No evidence of difference in mean TAT.")

    # ---- Per-flight breakdown ----
    if flight_stats:
        a("")
        a("PER-FLIGHT TAT BREAKDOWN  (pooled across all seeds × flights)")
        a(SEP_THIN)
        a(f"  {'Condition':<22} {'N flights':>10} {'Mean TAT':>10} {'Median':>10} {'SD':>8} "
          f"{'Min':>8} {'Max':>8} {'Wait unload':>12} {'Wait load':>10}")
        a(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
        for cond in ["gas", "electric"]:
            s = flight_stats.get(cond)
            if s:
                label = "Gas (conventional)" if cond == "gas" else "Electric"
                a(f"  {label:<22} {s['n_flights']:>10} {_f(s['mean_tat']):>10} "
                  f"{_f(s['median_tat']):>10} {_f(s['std_tat']):>8} "
                  f"{_f(s['min_tat']):>8} {_f(s['max_tat']):>8} "
                  f"{_f(s['mean_wait_unload']):>12} {_f(s['mean_wait_load']):>10}")

    # ---- GSE energy ----
    if gse_stats:
        a("")
        a("GSE ENERGY & TASK LOAD  (pooled across all seeds × GSEs)")
        a(SEP_THIN)
        a(f"  {'Condition':<22} {'Mean energy %':>14} {'SD energy':>10} {'Mean tasks':>12} {'SD tasks':>10}")
        a(f"  {'-'*22} {'-'*14} {'-'*10} {'-'*12} {'-'*10}")
        for cond in ["gas", "electric"]:
            s = gse_stats.get(cond)
            if s:
                label = "Gas (conventional)" if cond == "gas" else "Electric"
                a(f"  {label:<22} {_f(s['mean_energy']):>14} {_f(s['std_energy']):>10} "
                  f"{_f(s['mean_tasks']):>12} {_f(s['std_tasks']):>10}")

    # ---- Seed-level table ----
    a("")
    a("SEED-LEVEL TAT  (first 20 rows shown; full data in batch_results.csv)")
    a(SEP_THIN)
    a(f"  {'Seed':>5}  {'Gas TAT':>10}  {'Elec TAT':>10}  {'Diff (G-E)':>12}  "
      f"{'N gas':>7}  {'N elec':>7}")
    a(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*7}  {'-'*7}")
    display_df = df_complete.head(20)
    for _, row in display_df.iterrows():
        d_row = row["tat_conventional"] - row["tat_electric"]
        sign = "+" if d_row >= 0 else ""
        a(f"  {int(row['seed']):>5}  {_f(row['tat_conventional']):>10}  "
          f"{_f(row['tat_electric']):>10}  {sign + _f(d_row):>12}  "
          f"{int(row['n_completed_conv']):>7}  {int(row['n_completed_elec']):>7}")

    if len(df_complete) > 20:
        a(f"  ... ({len(df_complete) - 20} more rows in {csv_path.name})")

    a("")
    a(SEP_THICK)
    a("END OF REPORT")
    a(SEP_THICK)

    return "\n".join(lines)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 1 — Paired t-test analysis")
    parser.add_argument("--batch", type=str, default=None, help="Path to batch folder")
    parser.add_argument("--csv",   type=str, default=None, help="Path to batch_results.csv directly")
    args = parser.parse_args()

    try:
        csv_path, batch_dir = _resolve_csv(args.batch, args.csv)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} does not exist. Run batch_run.py first.", file=sys.stderr)
        sys.exit(1)

    report = analyse(csv_path, batch_dir)
    print(report)

    ts          = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = batch_dir / f"experiment_1_report_{ts}.txt"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
