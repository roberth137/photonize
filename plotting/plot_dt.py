import numpy as np
import matplotlib.pyplot as plt
plt.ion() # turn on interactive mode
import get_photons
import plotting as _p

def hist_dt_event(i):
    this_event = _p.group_events.iloc[i]

    this_event_photons = get_photons.crop_event(this_event, _p.pick_photons, _p.diameter)

    bin_size = 5
    bins = np.arange(min(this_event_photons.dt), max(this_event_photons.dt) + bin_size, bin_size)
    plt.figure(figsize=(8, 6))
    plt.hist(this_event_photons['dt'], bins=bins)
    plt.plot([], [], ' ', label=f'Total number of photons: {len(this_event_photons)}')
    plt.plot([], [], ' ', label=f'Lifetime: {this_event.lifetime:.3f}')
    plt.title("Histogram of dt values")
    plt.xlabel("arrival time ")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')  # Adjust the legend position if needed
    #plt.savefig(f"plot_dt_{i}.png")


if __name__ == "__main__":
    # Plot the histogram for the first event in _p.group_events
    plt.ioff()
    hist_dt_event(1)
    plt.show(block=True)