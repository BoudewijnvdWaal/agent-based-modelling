"""
find_n_seeds_cv.py — Approach 2: stabilization of the coefficient of variation
to determine the minimum number of simulation runs needed.

CV = σ(o) / μ(o)

Stabilization is detected when the CV has not changed by more than `eps`
(relative change) over the last `window` consecutive runs.

Tracks three outputs:
  - mean_tat_gas       (raw per-seed TAT for conventional GSE)
  - mean_tat_electric  (raw per-seed TAT for electric GSE)
  - diff               (paired difference: gas − electric)

Usage:
    python find_n_seeds_cv.py [--window 20] [--eps 0.02] [--max-runs 300]
    python find_n_seeds_cv.py --output cv_plot.png
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOG_DIR      = PROJECT_ROOT / "run logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from batch_run import parse_log, _silence


def _run_paired_seed(seed: int, batch_dir: Path) -> dict:
    """Run both gas and electric conditions for one seed with the same spawn config."""
    import run_me

    log_gas  = batch_dir / f"seed{seed:03d}_gas.log"
    log_elec = batch_dir / f"seed{seed:03d}_electric.log"

    spawn_cfg = run_me.build_random_gse_spawn_config(
        run_me.nodes_dict, run_me.GSE_COUNT, seed=seed
    )
    with _silence():
        run_me.run_simulation(
            2.0, 0.25, "gas",
            log_filepath=log_gas, gse_spawn_config=spawn_cfg,
            enable_visualization=False, pace=False, show_status=False,
        )
    with _silence():
        run_me.run_simulation(
            15.0, 0.5, "electric",
            log_filepath=log_elec, gse_spawn_config=spawn_cfg,
            enable_visualization=False, pace=False, show_status=False,
        )

    return {
        "tat_gas":      parse_log(log_gas)["mean_tat"],
        "tat_electric": parse_log(log_elec)["mean_tat"],
    }

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
WINDOW   = 20     # number of consecutive runs over which CV must be stable
EPS      = 0.02   # max relative change in CV to be considered stable (2%)
MIN_RUNS = 30     # minimum runs before checking stabilization
MAX_RUNS = 300    # hard cap
# =============================================================================


def _cv(values: list[float]) -> float:
    """Coefficient of variation: std / mean. Returns inf if mean ≈ 0."""
    if len(values) < 2:
        return float("inf")
    mu = sum(values) / len(values)
    if abs(mu) < 1e-9:
        return float("inf")
    variance = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / abs(mu)


def _stable(cv_history: list[float], window: int, eps: float) -> bool:
    """True when the last `window` CV values all fall within eps of their mean."""
    if len(cv_history) < window:
        return False
    recent = cv_history[-window:]
    mu = sum(recent) / len(recent)
    if abs(mu) < 1e-9:
        return False
    return all(abs(v - mu) / abs(mu) <= eps for v in recent)


def find_n_seeds_cv(
    window: int = WINDOW,
    eps: float = EPS,
    min_runs: int = MIN_RUNS,
    max_runs: int = MAX_RUNS,
) -> dict:
    """
    Run paired simulations and track CV stabilization for three outputs.
    Returns full history and the stabilization point n* for each output.
    """
    batch_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = LOG_DIR / f"find_n_cv_{batch_ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("Approach 2 — Coefficient of Variation stabilization")
    print(f"  stabilization window = {window} runs,  ε = {eps*100:.1f}%")
    print(f"  min runs = {min_runs},  max runs = {max_runs}")
    print(f"  logs → {batch_dir}\n")
    print(f"  {'k':>4}  {'TAT_gas':>9}  {'TAT_elec':>9}  {'diff':>8}  "
          f"{'CV_gas':>8}  {'CV_elec':>8}  {'CV_diff':>8}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    vals_gas  = []
    vals_elec = []
    vals_diff = []

    cv_gas_hist  = []
    cv_elec_hist = []
    cv_diff_hist = []

    history  = []
    n_stable = {"gas": None, "electric": None, "diff": None}
    k_valid  = 0

    for seed in range(max_runs):
        result   = _run_paired_seed(seed, batch_dir)
        tat_gas  = result["tat_gas"]
        tat_elec = result["tat_electric"]

        if tat_gas is None or tat_elec is None:
            print(f"  seed {seed:3d} — skipped (incomplete run)")
            continue

        k_valid += 1
        vals_gas.append(tat_gas)
        vals_elec.append(tat_elec)
        vals_diff.append(tat_gas - tat_elec)

        cv_g = _cv(vals_gas)
        cv_e = _cv(vals_elec)
        cv_d = _cv(vals_diff)

        cv_gas_hist.append(cv_g)
        cv_elec_hist.append(cv_e)
        cv_diff_hist.append(cv_d)

        stable_gas  = k_valid >= min_runs and _stable(cv_gas_hist,  window, eps)
        stable_elec = k_valid >= min_runs and _stable(cv_elec_hist, window, eps)
        stable_diff = k_valid >= min_runs and _stable(cv_diff_hist, window, eps)

        if stable_gas  and n_stable["gas"]      is None: n_stable["gas"]      = k_valid
        if stable_elec and n_stable["electric"] is None: n_stable["electric"] = k_valid
        if stable_diff and n_stable["diff"]     is None: n_stable["diff"]     = k_valid

        markers = "".join([
            " ✓gas"  if stable_gas  and n_stable["gas"]      == k_valid else "",
            " ✓elec" if stable_elec and n_stable["electric"] == k_valid else "",
            " ✓diff" if stable_diff and n_stable["diff"]     == k_valid else "",
        ])

        print(
            f"  {k_valid:>4}  {tat_gas:>9.2f}  {tat_elec:>9.2f}  "
            f"{tat_gas - tat_elec:>8.2f}  {cv_g:>8.4f}  {cv_e:>8.4f}  {cv_d:>8.4f}"
            + markers
        )

        history.append({
            "k":        k_valid,
            "seed":     seed,
            "tat_gas":  tat_gas,
            "tat_elec": tat_elec,
            "diff":     tat_gas - tat_elec,
            "cv_gas":   cv_g,
            "cv_elec":  cv_e,
            "cv_diff":  cv_d,
        })

        if all(v is not None for v in n_stable.values()):
            print(f"\n  ✓ All three outputs stabilized.")
            break

    # fill any that never stabilized
    for key in n_stable:
        if n_stable[key] is None:
            n_stable[key] = k_valid
            print(f"  ! '{key}' CV did not stabilize within {max_runs} runs — using {k_valid}")

    n_recommended = max(n_stable.values())
    print(f"\n  → Recommended N_SEEDS = {n_recommended}  "
          f"(max of gas={n_stable['gas']}, elec={n_stable['electric']}, diff={n_stable['diff']})\n")

    return {
        "history":       history,
        "n_stable":      n_stable,
        "n_recommended": n_recommended,
        "window":        window,
        "eps":           eps,
        "min_runs":      min_runs,
        "batch_dir":     batch_dir,
    }


def plot_results(result: dict, save_path: Path = None) -> None:
    hist = result["history"]
    ks       = [h["k"]       for h in hist]
    cv_gas   = [h["cv_gas"]  for h in hist]
    cv_elec  = [h["cv_elec"] for h in hist]
    cv_diff  = [h["cv_diff"] for h in hist]

    n_gas  = result["n_stable"]["gas"]
    n_elec = result["n_stable"]["electric"]
    n_diff = result["n_stable"]["diff"]
    n_rec  = result["n_recommended"]
    window = result["window"]
    eps    = result["eps"]

    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                             constrained_layout=True)

    fig.suptitle(
        "Approach 2 — Coefficient of Variation Stabilization\n"
        f"Stabilization: window = {window} runs, ε = {eps*100:.0f}%  |  "
        f"Recommended N* = {n_rec}",
        fontsize=11, fontweight="bold",
    )

    configs = [
        (axes[0], cv_gas,  n_gas,  "#2E86AB", "CV — Mean TAT (gas GSE)",      "gas"),
        (axes[1], cv_elec, n_elec, "#2E8B57", "CV — Mean TAT (electric GSE)", "electric"),
        (axes[2], cv_diff, n_diff, "#8B2E86", "CV — Paired TAT difference",   "diff"),
    ]

    for ax, cv_vals, n_stab, color, title, label in configs:
        ax.scatter(ks, cv_vals, color=color, alpha=0.45, s=18, zorder=2)
        ax.plot(ks, cv_vals, color=color, lw=1.5, zorder=3, label=r"$c_v = \sigma / \mu$")

        # rolling mean of CV to show trend
        if len(cv_vals) >= window:
            rolling = np.convolve(cv_vals, np.ones(window) / window, mode="valid")
            rolling_ks = ks[window - 1:]
            ax.plot(rolling_ks, rolling, color=color, lw=2.5, ls="--",
                    alpha=0.7, label=f"{window}-run rolling mean")

        ax.axvline(n_stab, color="#C0392B", lw=1.5, ls="--",
                   label=f"Stable at k = {n_stab}")

        # shade stable region
        ax.axvspan(n_stab, max(ks), alpha=0.07, color="#C0392B")

        ax.set_ylabel(r"$c_v$", fontsize=10)
        ax.set_title(title, fontsize=10, pad=5)
        ax.legend(fontsize=8, framealpha=0.8, loc="upper right")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[2].set_xlabel("Number of simulation runs  k", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved → {save_path}")
    else:
        plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find minimum simulation runs via CV stabilization (Approach 2)."
    )
    parser.add_argument("--window",   type=int,   default=WINDOW,
                        help=f"Stabilization window in runs (default: {WINDOW})")
    parser.add_argument("--eps",      type=float, default=EPS,
                        help=f"Max relative CV change for stability (default: {EPS})")
    parser.add_argument("--min-runs", type=int,   default=MIN_RUNS,
                        help=f"Minimum runs before checking (default: {MIN_RUNS})")
    parser.add_argument("--max-runs", type=int,   default=MAX_RUNS,
                        help=f"Hard cap on runs (default: {MAX_RUNS})")
    parser.add_argument("--output",   type=str,   default=None,
                        help="Save figure to this path (e.g. cv_plot.png)")
    args = parser.parse_args()

    result = find_n_seeds_cv(
        window=args.window,
        eps=args.eps,
        min_runs=args.min_runs,
        max_runs=args.max_runs,
    )

    save_path = Path(args.output) if args.output else None
    plot_results(result, save_path=save_path)
