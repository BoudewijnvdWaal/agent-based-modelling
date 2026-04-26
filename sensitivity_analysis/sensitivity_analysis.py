"""
Sensitivity Analysis for GSE Fleet Size and Auction Parameters
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add baseline_code to path
BASE_DIR = Path(__file__).resolve().parent.parent / "baseline_code"
sys.path.insert(0, str(BASE_DIR))

from run_me import run_simulation, build_random_gse_spawn_config, nodes_dict

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
N_RUNS = 114  # number of seeds to average over per config
DEFAULT_GSE_SPEED = 4.0
DEFAULT_AUCTION_ALPHA = 0.7
DEFAULT_AUCTION_BETA = 0.3
DEFAULT_CHARGE_DURATION = 15.0
DEFAULT_CONSUMPTION_RATE = 0.5
DEFAULT_GSE_TYPE = "electric"

# =============================================================================
# HELPER: run N seeds and return averaged metrics
# =============================================================================
def single_run_for_parallel(gse_count, auction_alpha, auction_beta, seed):
    """Helper function for parallel execution"""
    spawn_config = build_random_gse_spawn_config(nodes_dict, gse_count, seed=seed)
    return run_simulation(
        gse_count=gse_count,
        gse_speed=DEFAULT_GSE_SPEED,
        auction_alpha=auction_alpha,
        auction_beta=auction_beta,
        charge_duration=DEFAULT_CHARGE_DURATION,
        base_consumption_rate=DEFAULT_CONSUMPTION_RATE,
        gse_type_label=DEFAULT_GSE_TYPE,
        simulation_duration_hours=24,
        gse_spawn_config=spawn_config,
        enable_visualization=False,
        pace=False,
        show_status=False,
    )

def run_averaged(gse_count, auction_alpha, auction_beta, n_runs=N_RUNS):
    """
    Runs the simulation n_runs times with different random GSE spawn seeds
    and returns the mean of each metric across runs.
    Uses parallel execution for speed.
    """
    tat_values = []
    all_metrics = []
    with ProcessPoolExecutor(max_workers=min(n_runs, os.cpu_count() or 4)) as executor:
        futures = [executor.submit(single_run_for_parallel, gse_count, auction_alpha, auction_beta, seed) for seed in range(n_runs)]
        for future in as_completed(futures):
            metrics = future.result()
            all_metrics.append(metrics)
            if metrics['mean_tat'] is not None:
                tat_values.extend(metrics['tat_values'])

    return {
        'mean_tat': np.mean(tat_values) if tat_values else None,
        'median_tat': np.median(tat_values) if tat_values else None,
        'std_tat': np.std(tat_values) if tat_values else None,
        'min_tat': np.min(tat_values) if tat_values else None,
        'max_tat': np.max(tat_values) if tat_values else None,
        'tat_values': tat_values,
        'n_completed': np.mean([m['n_completed'] for m in all_metrics]),
        'total_energy_used': np.mean([m['total_energy_used'] for m in all_metrics]),
        'avg_tasks_per_gse': np.mean([m['avg_tasks_per_gse'] for m in all_metrics]),
    }

# =============================================================================
# 1. GSE COUNT SENSITIVITY
# =============================================================================
def run_sensitivity_gse_count(gse_counts, n_runs=N_RUNS):
    results = []
    for count in gse_counts:
        print(f"[GSE Count] count={count}, averaging over {n_runs} seeds...")
        m = run_averaged(gse_count=count, auction_alpha=DEFAULT_AUCTION_ALPHA, auction_beta=DEFAULT_AUCTION_BETA, n_runs=n_runs)
        results.append({'gse_count': count, **m})
        print(f"  -> mean_TAT={m['mean_tat']:.1f} min (std={m['std_tat']:.1f}), median={m['median_tat']:.1f}, range=[{m['min_tat']:.1f}, {m['max_tat']:.1f}], tasks/GSE={m['avg_tasks_per_gse']:.1f}, energy={m['total_energy_used']:.1f}%")
    return pd.DataFrame(results)

def plot_gse_count_sensitivity(df, output_dir="."):
    output_dir = Path(output_dir)
    
    # --- Main plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GSE Fleet Size Sensitivity Analysis", fontsize=14, fontweight='bold')

    # TAT with std band
    ax = axes[0, 0]
    ax.plot(df['gse_count'], df['mean_tat'], 'o-', color='steelblue', label='Mean TAT')
    ax.fill_between(df['gse_count'], df['mean_tat'] - df['std_tat'], df['mean_tat'] + df['std_tat'], alpha=0.2, color='steelblue', label='�1 std')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Mean Turnaround Time (min)')
    ax.set_title('TAT vs GSE Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tasks per GSE
    ax = axes[0, 1]
    ax.plot(df['gse_count'], df['avg_tasks_per_gse'], 's-', color='darkorange')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Average Tasks per GSE')
    ax.set_title('GSE Utilization (Tasks per GSE)')
    ax.grid(True, alpha=0.3)

    # Total energy
    ax = axes[1, 0]
    ax.plot(df['gse_count'], df['total_energy_used'], '^-', color='green')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Total Energy Used (cumulative %)')
    ax.set_title('Total Fleet Energy Consumption')
    ax.grid(True, alpha=0.3)

    # Completed flights
    ax = axes[1, 1]
    ax.plot(df['gse_count'], df['n_completed'], 'D-', color='firebrick')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Number of Completed Flights')
    ax.set_title('Completed Flights vs GSE Count')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'sensitivity_gse_count.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out}")

    # --- TAT Boxplot ---
    tat_lists = df['tat_values'].tolist()
    labels = [str(int(c)) for c in df['gse_count']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(tat_lists, labels=labels)
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Turnaround Time (min)')
    ax.set_title('TAT Distribution per GSE Count (Boxplot)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / 'tat_boxplot_gse_count.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {out}")

# =============================================================================
# 2. ALPHA / BETA SENSITIVITY (COMMENTED OUT)
# =============================================================================
# def run_sensitivity_alpha_beta(alpha_values, beta_values, n_runs=N_RUNS):
#     results = []
#     total = len(alpha_values) * len(beta_values)
#     done = 0
#     for alpha in alpha_values:
#         for beta in beta_values:
#             done += 1
#             print(f"[Alpha/Beta] ({done}/{total}) alpha={alpha:.2f}, beta={beta:.2f}")
#             m = run_averaged(gse_count=8, auction_alpha=alpha, auction_beta=beta, n_runs=n_runs)
#             results.append({'alpha': round(alpha, 2), 'beta': round(beta, 2), **m})
#             print(f"  -> mean_TAT={m['mean_tat']:.1f} min, median={m['median_tat']:.1f}, range=[{m['min_tat']:.1f}, {m['max_tat']:.1f}], tasks/GSE={m['avg_tasks_per_gse']:.1f}")
#     return pd.DataFrame(results)

# def plot_alpha_beta_sensitivity(df, output_dir="."):
#     # Create 2D heatmap for TAT
#     pivot_tat = df.pivot(index='beta', columns='alpha', values='mean_tat')
#     plt.figure(figsize=(8, 6))
#     plt.imshow(pivot_tat, cmap='viridis', aspect='auto', origin='lower')
#     plt.colorbar(label='Mean TAT (min)')
#     plt.xlabel('Alpha')
#     plt.ylabel('Beta')
#     plt.title('TAT Heatmap (Alpha vs Beta)')
#     plt.xticks(range(len(pivot_tat.columns)), [f'{x:.1f}' for x in pivot_tat.columns])
#     plt.yticks(range(len(pivot_tat.index)), [f'{x:.1f}' for x in pivot_tat.index])
#     out = Path(output_dir) / 'sensitivity_alpha_beta_heatmap.png'
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"[Plot] Saved: {out}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "sensitivity_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. GSE Count sensitivity ---
    print("\n" + "=" * 60)
    print("  PART 1: GSE COUNT SENSITIVITY")
    print("=" * 60)
    gse_counts = list(range(4, 13))  # 4 to 12
    df_gse = run_sensitivity_gse_count(gse_counts, n_runs=N_RUNS)
    df_gse.to_csv(output_dir / 'sensitivity_gse_count.csv', index=False)
    plot_gse_count_sensitivity(df_gse, output_dir=output_dir)
    print(df_gse.to_string(index=False))

    # --- 2. Alpha / Beta sensitivity (COMMENTED OUT) ---
    # print("\n" + "=" * 60)
    # print("  PART 2: ALPHA / BETA SENSITIVITY")
    # print("=" * 60)
    # alpha_values = np.round(np.arange(0.1, 1.01, 0.1), 2)
    # beta_values = np.round(np.arange(0.1, 1.01, 0.1), 2)
    # df_ab = run_sensitivity_alpha_beta(alpha_values, beta_values, n_runs=N_RUNS)
    # df_ab.to_csv(output_dir / 'sensitivity_alpha_beta.csv', index=False)
    # plot_alpha_beta_sensitivity(df_ab, output_dir=output_dir)
    # print(df_ab.to_string(index=False))

    print("\n[Done] All results saved to:", output_dir)
