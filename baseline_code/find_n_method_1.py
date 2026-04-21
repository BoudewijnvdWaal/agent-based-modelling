"""
find_n_seeds.py — Sequential stopping rule (Approach 1) to determine the
minimum number of paired simulation runs needed for reliable analysis.

Stopping criterion:  2 * z_{α/2} * S / √k  <  l
Output metric:       paired difference  mean_tat_gas − mean_tat_electric  (minutes)

Usage:
    python find_n_seeds.py [--alpha 0.05] [--l 5.0] [--min-runs 30] [--max-runs 200]
    python find_n_seeds.py --output n_seeds_plot.png
"""

import argparse
import math
import os
import sys
import tempfile
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
from batch_run import _run_seed  # reuse the paired-run worker

# =============================================================================
# DEFAULT PARAMETERS  (edit here or pass via CLI)
# =============================================================================
ALPHA    = 0.05   # significance level  →  z ≈ 1.96
L        = 25.0    # desired full CI width in minutes (stop when CI < L)
MIN_RUNS = 30     # practical minimum before checking criterion
MAX_RUNS = 200    # hard cap to prevent infinite loop
# =============================================================================


def _z(alpha: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(1 - alpha / 2))
    except ImportError:
        table = {0.10: 1.645, 0.05: 1.960, 0.03: 2.170, 0.01: 2.576}
        if alpha in table:
            return table[alpha]
        raise ImportError(f"scipy not found and alpha={alpha} not in lookup table")


def find_n_seeds(
    alpha: float = ALPHA,
    l: float = L,
    min_runs: int = MIN_RUNS,
    max_runs: int = MAX_RUNS,
) -> dict:
    """
    Run paired simulations one seed at a time and apply the slide's
    sequential stopping rule to the paired TAT difference.

    Update formulas (from lecture slides):
        X̄_{j+1} = X̄_j + (X_{j+1} - X̄_j) / (j+1)
        S²_{j+1} = (1 - 1/j) * S²_j + (j+1) * (X̄_{j+1} - X̄_j)²

    Returns a dict with the full history and the found n*.
    """
    z = _z(alpha)
    batch_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = LOG_DIR / f"find_n_{batch_ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sequential stopping rule")
    print(f"  α = {alpha}  →  z = {z:.4f}")
    print(f"  desired CI width  l = {l} min")
    print(f"  min runs = {min_runs},  max runs = {max_runs}")
    print(f"  logs → {batch_dir}\n")
    print(f"  {'k':>4}  {'diff':>8}  {'x̄':>8}  {'S':>8}  {'CI width':>10}  {'stop?':>6}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")

    xbar    = 0.0
    s2      = 0.0
    history = []
    n_found = None
    k_valid = 0  # counts only runs with valid TAT output

    for seed in range(max_runs):
        result   = _run_seed(seed, str(batch_dir))
        tat_gas  = result["tat_conventional"]
        tat_elec = result["tat_electric"]

        if tat_gas is None or tat_elec is None:
            print(f"  seed {seed:3d} — skipped (incomplete run)")
            continue

        k_valid += 1
        diff     = tat_gas - tat_elec

        # --- update formulas from slides ---
        xbar_new = xbar + (diff - xbar) / k_valid
        if k_valid >= 2:
            s2 = (1 - 1 / (k_valid - 1)) * s2 + k_valid * (xbar_new - xbar) ** 2
        xbar = xbar_new

        s          = math.sqrt(s2) if s2 > 0 else 0.0
        ci_width   = 2 * z * s / math.sqrt(k_valid) if k_valid >= 2 else float("inf")
        ci_half    = ci_width / 2
        converged  = k_valid >= min_runs and ci_width < l

        history.append({
            "k":        k_valid,
            "seed":     seed,
            "diff":     diff,
            "xbar":     xbar,
            "s":        s,
            "ci_width": ci_width,
            "ci_lo":    xbar - ci_half,
            "ci_hi":    xbar + ci_half,
        })

        stop_marker = " ✓ STOP" if converged and n_found is None else ""
        print(
            f"  {k_valid:>4}  {diff:>8.2f}  {xbar:>8.2f}  {s:>8.2f}"
            f"  {ci_width:>10.3f}{stop_marker}"
        )

        if converged and n_found is None:
            n_found = k_valid
            break

    if n_found is None:
        n_found = k_valid
        print(f"\n  ! Criterion not met within {max_runs} seeds — using max: {n_found}")
    else:
        print(f"\n  ✓ Stopping criterion met at k = {n_found}")

    print(f"\n  → Set N_SEEDS = {n_found} in batch_run.py\n")

    return {
        "n_found":   n_found,
        "history":   history,
        "alpha":     alpha,
        "l":         l,
        "z":         z,
        "min_runs":  min_runs,
        "batch_dir": batch_dir,
    }


def plot_results(result: dict, save_path: Path = None) -> None:
    hist = result["history"]
    ks        = [h["k"]        for h in hist]
    xbars     = [h["xbar"]     for h in hist]
    ci_los    = [h["ci_lo"]    for h in hist]
    ci_his    = [h["ci_hi"]    for h in hist]
    diffs     = [h["diff"]     for h in hist]
    ci_widths = [h["ci_width"] for h in hist if h["ci_width"] < float("inf")]
    ks_valid  = [h["k"]        for h in hist if h["ci_width"] < float("inf")]

    n     = result["n_found"]
    l     = result["l"]
    alpha = result["alpha"]
    z     = result["z"]
    conf  = int((1 - alpha) * 100)

    # --- style ---
    plt.rcParams.update({
        "font.family":  "serif",
        "font.size":    10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"hspace": 0.35}
    )

    fig.suptitle(
        "Sequential Stopping Rule — Number of Simulation Runs Required\n"
        f"Output: paired TAT difference  (gas − electric)  |  "
        f"α = {alpha},  l = {l} min,  n* = {n}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    # ── Panel 1: running mean + CI ──────────────────────────────────────────
    ax1.scatter(ks, diffs, color="#4C8BBF", alpha=0.4, s=20, zorder=2,
                label="Per-seed difference")
    ax1.plot(ks, xbars, color="#1F5C8B", lw=2, zorder=3,
             label=r"Running mean $\bar{X}_k$")
    ax1.fill_between(ks, ci_los, ci_his, alpha=0.18, color="#4C8BBF",
                     label=f"{conf}% confidence interval")
    ax1.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax1.axvline(n, color="#C0392B", lw=1.5, ls="--", zorder=4,
                label=f"n* = {n}")
    ax1.set_ylabel("TAT difference (min)", fontsize=10)
    ax1.set_title("Running mean of paired TAT difference with confidence interval",
                  fontsize=10, pad=6)
    ax1.legend(fontsize=9, framealpha=0.8, loc="upper right")
    ax1.grid(True, alpha=0.25, linewidth=0.5)

    # ── Panel 2: CI width convergence ───────────────────────────────────────
    ax2.plot(ks_valid, ci_widths, color="#E07B20", lw=2,
             label=r"CI width  $2 z_{\alpha/2} S / \sqrt{k}$")
    ax2.axhline(l, color="#C0392B", lw=1.5, ls="--",
                label=f"Threshold  l = {l} min")
    ax2.axvline(n, color="#C0392B", lw=1.5, ls="--",
                label=f"n* = {n}")

    # shade convergence region
    ax2.fill_between(ks_valid, ci_widths, l,
                     where=[cw < l for cw in ci_widths],
                     alpha=0.15, color="#C0392B",
                     label="Criterion satisfied")

    ax2.set_ylabel("CI width (min)", fontsize=10)
    ax2.set_xlabel("Number of simulation runs  k", fontsize=10)
    ax2.set_title(
        r"Convergence of confidence interval width  ($2 z_{\alpha/2} S / \sqrt{k}$)",
        fontsize=10, pad=6,
    )
    ax2.legend(fontsize=9, framealpha=0.8, loc="upper right")
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # annotation at n*
    for ax in (ax1, ax2):
        ax.annotate(
            f" n* = {n}",
            xy=(n, ax.get_ylim()[0]),
            xytext=(n + 0.5, ax.get_ylim()[0]),
            fontsize=9, color="#C0392B", va="bottom",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

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
        description="Find the minimum number of simulation runs via Approach 1 sequential stopping."
    )
    parser.add_argument("--alpha",    type=float, default=ALPHA,
                        help=f"Significance level (default: {ALPHA})")
    parser.add_argument("--l",        type=float, default=L,
                        help=f"Desired CI width in minutes (default: {L})")
    parser.add_argument("--min-runs", type=int,   default=MIN_RUNS,
                        help=f"Minimum runs before checking criterion (default: {MIN_RUNS})")
    parser.add_argument("--max-runs", type=int,   default=MAX_RUNS,
                        help=f"Hard cap on number of runs (default: {MAX_RUNS})")
    parser.add_argument("--output",   type=str,   default=None,
                        help="Save figure to this file path (e.g. n_seeds.png)")
    args = parser.parse_args()

    result = find_n_seeds(
        alpha=args.alpha,
        l=args.l,
        min_runs=args.min_runs,
        max_runs=args.max_runs,
    )

    save_path = Path(args.output) if args.output else None
    plot_results(result, save_path=save_path)
