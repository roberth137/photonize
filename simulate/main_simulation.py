#!/usr/bin/env python3
import numpy as np
import pandas as pd
import simulate_fit_events as sim
import matplotlib.pyplot as plt


def run_simulation(filename, diameters, method, n_events=10000, random_seed=42):
    """
    Run the simulate_and_fit_events for a range of diameters on one dataset and method.

    Returns:
        pandas.DataFrame with columns ['filename', 'method', 'diameter', 'mean_error']
    """
    # Load event statistics and limit to n_events for speed
    event_stats = pd.read_hdf(filename)
    event_stats = event_stats.iloc[:n_events]

    results = []
    for diameter in diameters:
        # Simulate and fit
        delta_x, delta_y, _ = sim.simulate_and_fit_events(
            event_stats,
            method=method,
            diameter=diameter,
            random_seed=random_seed
        )
        # Compute radial error
        error = np.hypot(delta_x, delta_y)
        mean_error = np.mean(error)

        results.append({
            'filename': filename,
            'method': method,
            'diameter': diameter,
            'mean_error': mean_error
        })

    return pd.DataFrame(results)


def main():
    # List your two datasets here
    filenames = [
        #"simulate/sim_experiments_stats/dset1.hdf5",
        #"simulate/sim_experiments_stats/dset2.hdf5",
        #"simulate/sim_experiments_stats/legacy.hdf5",
        "simulate/sim_experiments_stats/origami.hdf5"
    ]

    # Methods to compare
    methods = ['com', 'mle_fixed', 'mle', 'pass']

    # Range of diameters from 3 to 7 in 0.1 steps
    diameters = np.arange(2.0, 8 + 1e-6, 0.25)

    # Simulation settings
    n_events = 1000
    random_seed = 42

    # Collect results
    all_results = []
    for filename in filenames:
        for method in methods:
            df = run_simulation(
                filename,
                diameters,
                method=method,
                n_events=n_events,
                random_seed=random_seed
            )
            all_results.append(df)

    # Combine into one DataFrame
    all_results = pd.concat(all_results, ignore_index=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    for filename in filenames:
        for method in methods:
            subset = all_results[
                (all_results['filename'] == filename) &
                (all_results['method'] == method)
            ]
            label = f"{method.upper()} ({filename.split('/')[-1]})"
            plt.plot(
                subset['diameter'],
                subset['mean_error'],
                marker='o',
                label=label
            )

    plt.xlabel("Diameter")
    plt.ylabel("Mean Localization Error")
    plt.title("Mean Error vs Diameter\nfor COM vs MLE_FIXED across Datasets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
