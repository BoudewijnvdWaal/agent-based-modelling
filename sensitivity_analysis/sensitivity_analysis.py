import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent.parent / "baseline_code"
sys.path.insert(0, str(BASE_DIR))

from batch_run import _silence, _bar
with _silence():
    from run_me import run_simulation, build_random_gse_spawn_config, nodes_dict

# Default parameters
N_RUNS = 114
DEFAULT_GSE_SPEED = 4.0
DEFAULT_AUCTION_ALPHA = 0.7
DEFAULT_AUCTION_BETA = 0.3
DEFAULT_CHARGE_DURATION = 15.0
DEFAULT_CONSUMPTION_RATE = 0.5
DEFAULT_GSE_TYPE = "electric"


def single_run_for_parallel(gse_count, auction_alpha, auction_beta, seed):
    spawn_config = build_random_gse_spawn_config(nodes_dict, gse_count, seed=seed)
    with _silence():
        metrics = run_simulation(
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
        )
    # Tag with gse_count so results can be grouped after parallel execution
    metrics['gse_count'] = gse_count
    return metrics


# Submits all configs × seeds at once to keep all CPUs busy the entire run
def run_sensitivity_gse_count(gse_counts, n_runs=N_RUNS):
    total_jobs = len(gse_counts) * n_runs
    all_results = {count: [] for count in gse_counts}
    start = time.monotonic()
    done = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(single_run_for_parallel, count, DEFAULT_AUCTION_ALPHA, DEFAULT_AUCTION_BETA, seed)
            for count in gse_counts
            for seed in range(n_runs)
        ]
        for future in as_completed(futures):
            metrics = future.result()
            all_results[metrics['gse_count']].append(metrics)
            done += 1

            elapsed = time.monotonic() - start
            eta = (elapsed / done) * (total_jobs - done)
            configs_done = sum(1 for v in all_results.values() if len(v) == n_runs)
            sys.stdout.write(
                f"\r  {_bar(done, total_jobs)}  {done}/{total_jobs} seeds"
                f"  configs done: {configs_done}/{len(gse_counts)}"
                f"  elapsed: {timedelta(seconds=round(elapsed))}"
                f"  ETA: {timedelta(seconds=round(eta))}"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")

    rows = []
    for count in gse_counts:
        metrics_list = all_results[count]
        tat_values = [v for m in metrics_list if m['mean_tat'] is not None for v in m['tat_values']]
        rows.append({
            'gse_count':         count,
            'mean_tat':          np.mean(tat_values) if tat_values else None,
            'median_tat':        np.median(tat_values) if tat_values else None,
            'std_tat':           np.std(tat_values) if tat_values else None,
            'min_tat':           np.min(tat_values) if tat_values else None,
            'max_tat':           np.max(tat_values) if tat_values else None,
            'tat_values':        tat_values,
            'n_completed':       np.mean([m['n_completed'] for m in metrics_list]),
            'total_energy_used': np.mean([m['total_energy_used'] for m in metrics_list]),
            'avg_tasks_per_gse': np.mean([m['avg_tasks_per_gse'] for m in metrics_list]),
        })
    return pd.DataFrame(rows)


def plot_gse_count_sensitivity(df, output_dir="."):
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GSE Fleet Size Sensitivity Analysis", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(df['gse_count'], df['mean_tat'], 'o-', color='steelblue', label='Mean TAT')
    ax.fill_between(df['gse_count'], df['mean_tat'] - df['std_tat'], df['mean_tat'] + df['std_tat'],
                    alpha=0.2, color='steelblue', label='±1 std')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Mean Turnaround Time (min)')
    ax.set_title('TAT vs GSE Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df['gse_count'], df['avg_tasks_per_gse'], 's-', color='darkorange')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Average Tasks per GSE')
    ax.set_title('GSE Utilization (Tasks per GSE)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df['gse_count'], df['total_energy_used'], '^-', color='green')
    ax.set_xlabel('Number of GSEs')
    ax.set_ylabel('Total Energy Used (cumulative %)')
    ax.set_title('Total Fleet Energy Consumption')
    ax.grid(True, alpha=0.3)

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
    print(f"Saved: {out}")

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
    print(f"Saved: {out}")


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "sensitivity_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    gse_counts = list(range(4, 20))
    print(f"Running {len(gse_counts)} configs × {N_RUNS} seeds = {len(gse_counts)*N_RUNS} total jobs on {os.cpu_count()} CPUs\n")
    df_gse = run_sensitivity_gse_count(gse_counts, n_runs=N_RUNS)
    df_gse.to_csv(output_dir / 'sensitivity_gse_count.csv', index=False)
    plot_gse_count_sensitivity(df_gse, output_dir=output_dir)
    print("\n", df_gse[['gse_count','mean_tat','std_tat','avg_tasks_per_gse']].to_string(index=False))
    print("\nDone →", output_dir)
