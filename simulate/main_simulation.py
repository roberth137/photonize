import numpy as np
import pandas as pd
import simulate_fit_events as sim

def main():
    filename = "simulate/sim_experiments_stats/2green_delta_0p3.hdf5"
    diameter = 4
    random_seed = 42
    method = 'com' # 'com' , 'mle' , 'mle_fixed'
    n_events = 10000 # limit for speeed
    # 1. Obtain event parameters from a file or generate them.
    #    Example: read from HDF5 file
    event_stats = pd.read_hdf(filename)
    event_stats_lim = event_stats[:n_events]

    delta_x, delta_y, _ = sim.simulate_and_fit_events(event_stats_lim,
                                                       method=method,
                                                       diameter=diameter,
                                                       random_seed=random_seed)

    error = np.hypot(delta_x, delta_y)

    # Print statistics
    print(f"Results from {(len(event_stats_lim)/1000)}k events:")
    print(f"Analyzed with method {method.upper()} and diameter {diameter}")
    print(f"Mean error: {np.mean(error)}")

    sim.plot_results(error, method=method, diameter=diameter)


if __name__ == '__main__':
    main()