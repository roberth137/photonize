#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simulate as s   # adjust this to however you import your simulate module

def simulate_and_fit_events_with_true_bg(
    event_stats: pd.DataFrame,
    method: str = 'com',
    diameter: float = s.fitting_diameter,
    random_seed: int = 42
):
    """
    Wrapper around your simulate_and_fit_events that also returns the true bg_counts.
    Returns:
        x_fit, y_fit      : np.ndarray
        bg_fit            : np.ndarray  (fitted background from analyze_sim_event)
        bg_true           : np.ndarray  (true background photon counts in ROI)
    """
    # copy‐paste your simulate_and_fit_events body up through bg_counts/population,
    # but return both bg_fit and bg_counts. For brevity I’ll call the original and
    # then re‐compute true bg_counts the same way:
    x_fit, y_fit, bg_fit = s.simulate_and_fit_events(event_stats,
                                                 method=method,
                                                 diameter=diameter,
                                                 random_seed=random_seed)

    bg_true = event_stats['bg'].astype(float).values

    for i in range(10):
        print(f'{bg_fit[i], bg_true[i]}')

    return x_fit, y_fit, bg_fit, bg_true

def test_bg(filename, diameters, method, n_events=10000, random_seed=42):
    """
    For a single file+method, run over diameters and return a DataFrame with:
       filename, method, diameter, mean_delta_bg, std_delta_bg
    """
    # load & trim
    event_stats = pd.read_hdf(filename).iloc[:n_events]

    rows = []
    for dia in diameters:
        x_fit, y_fit, bg_fit, bg_true = simulate_and_fit_events_with_true_bg(
            event_stats, method=method, diameter=dia, random_seed=random_seed
        )
        delta_bg = bg_fit - bg_true
        rows.append({
            'filename': filename,
            'method':   method,
            'diameter': dia,
            'mean_delta_bg': np.mean(delta_bg),
            'std_delta_bg':  np.std(delta_bg),
        })

    return pd.DataFrame(rows)

def main():
    # --- USER SETTINGS ---
    filenames = [
        "simulate/sim_experiments_stats/dset1.hdf5",
        "simulate/sim_experiments_stats/dset2.hdf5",
        "simulate/sim_experiments_stats/2green_0p3.hdf5",
        #"simulate/sim_experiments_stats/delta0p15.hdf5"
    ]
    methods   = ['mle']
    diameters = np.arange(2.0, 8.0 + 1e-6, 0.25)
    n_events  = 10
    random_seed = 42

    # run tests
    all_dfs = []
    for fn in filenames:
        for m in methods:
            df = test_bg(fn, diameters, m, n_events=n_events, random_seed=random_seed)
            all_dfs.append(df)
    results = pd.concat(all_dfs, ignore_index=True)

    # Plot: mean Δbg ± std vs diameter
    plt.figure(figsize=(10,6))
    for (fn, m), sub in results.groupby(['filename','method']):
        label = f"{m.upper()} ({fn.split('/')[-1]})"
        plt.errorbar(sub['diameter'], sub['mean_delta_bg'],
                     yerr=sub['std_delta_bg'], marker='o', linestyle='-',
                     label=label)
    plt.xlabel("Diameter")
    plt.ylabel("Mean ΔBackground (fitted – true)")
    plt.title("Background Fit Bias ± Std vs ROI Diameter")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
