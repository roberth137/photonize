import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simulate_events as sim

def main():
    filename = "simulate/sim_experiments_stats/2green_conditions.hdf5"
    diameter = 5
    # 1. Obtain event parameters from a file or generate them.
    #    Example: read from HDF5 file
    event_stats = pd.read_hdf(filename)

    dist_pure, dist_w_bg = sim.simulate_and_fit_events(event_stats, diameter)

    error_pure_mean, error_pure_std = np.mean(dist_pure), np.std(dist_pure)
    error_w_bg_mean, error_w_bg_std = np.mean(dist_w_bg), np.std(dist_w_bg)

    # Print statistics
    print(f"Results from {len(event_stats)} events:")
    print(f"No BG Correction - mean error: {error_pure_mean:.4f}, std: {error_pure_std:.4f}")
    print(f"With BG Correction - mean error: {error_w_bg_mean:.4f}, std: {error_w_bg_std:.4f}")

    sim.plot_results(dist_pure, dist_w_bg)


if __name__ == '__main__':
    main()