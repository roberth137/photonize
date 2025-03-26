# This is the main function for analysing a single event

import numpy as np
from fitting.on_off import get_on_off_dur
from fitting.localization import localize_com
from fitting.lifetime import avg_lifetime_weighted
from collections import namedtuple
import matplotlib.pyplot as plt

# Define the named tuple once
EventResult = namedtuple('EventResult', [
    'x_fit', 'y_fit', 'lifetime',
    'start_ms', 'end_ms', 'duration_ms',
    'num_photons'
])

def fit_event(photons, dt_peak, diameter):
    """
    Analyze a single photon event.

    Returns:
        EventResult namedtuple with:
        - x_fit, y_fit: fitted x/y position
        - lifetime: weighted average lifetime
        - start_ms, end_ms: event start/end time
        - duration_ms: event duration
        - num_photons: number of photons in the event
    """
    # get start and end of event
    start_ms, end_ms, duration_ms = get_on_off_dur(photons)

    # filter photons within the event time window
    event_photons = photons[(photons.ms >= start_ms) & (photons.ms <= end_ms)]

    # extract coordinates and arrival times
    x_photons = np.copy(event_photons.x)
    y_photons = np.copy(event_photons.y)
    dt_photons = np.copy(event_photons.dt)

    # fit x and y position using center of mass
    x_fit, y_fit, sdx, sdy = localize_com(
        event_photons.x, event_photons.y, return_sd=True
    )

    # calculate distance from center
    distances = np.sqrt((x_photons - x_fit) ** 2 + (y_photons - y_fit) ** 2)

    # calculate lifetime using weighted averaging
    lifetime = avg_lifetime_weighted(
        dt_photons, distance=distances, peak=dt_peak, diameter=diameter
    )

    # pack into a named tuple and return
    return EventResult(
        x_fit=x_fit,
        y_fit=y_fit,
        lifetime=lifetime,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        num_photons=len(event_photons)
    )


def simulate_event_ms_trace(seed=42, bg_rate=3, diameter=4.5, dur_all=400, dur_eve=300, brightness_eve=2):
    """
    Simulates a time trace (ms) with two components:
      - Background: uniformly distributed over 0 to dur_all.
      - Event: uniformly distributed over the central window (event_start to event_end).
    """
    np.random.seed(seed)
    # ROI area as given: (diameter/2)*pi which for diameter=4.5 gives 2.25*pi.
    area = (diameter / 2) * np.pi
    bg_photon_rate = bg_rate * area  # photons / 200ms over the ROI

    # 1. Simulate background photons
    n_bg_expected = int((dur_all / 200) * bg_photon_rate)
    bg_arrival_times = np.sort(np.random.uniform(0, dur_all, size=n_bg_expected))

    # 2. Simulate event photons
    event_start = (dur_all - dur_eve) / 2  # for dur_all=400 & dur_eve=300, event_start=50ms
    event_end = event_start + dur_eve  # event_end=350ms
    n_eve_expected = int(dur_eve * brightness_eve)
    event_arrival_times = np.sort(np.random.uniform(event_start, event_end, size=n_eve_expected))

    # Combine background and event arrival times
    all_arrival_times = np.sort(np.concatenate([bg_arrival_times, event_arrival_times]))
    return all_arrival_times

#test

if __name__ == "__main__":
    print('main loop')
    # 1. Generate ms time stamps using the simulation function
    ms_times = simulate_event_ms_trace(
        seed=42,
        bg_rate=3,
        diameter=4.5,
        dur_all=400,
        dur_eve=300,
        brightness_eve=2  # 2 photons per ms during the event
    )

    num_photons = len(ms_times)

    # 2. Generate spatial coordinates (x,y) for each photon.
    #    Each coordinate is normally distributed with mean 0 and std dev 1.1 (in pixels).
    x_coords = np.random.normal(loc=0, scale=1.1, size=num_photons)
    y_coords = np.random.normal(loc=0, scale=1.1, size=num_photons)

    # 3. Generate dt values: Exponentially distributed (decay parameter 280), shifted by 80 and capped at 2500.
    dt_vals = 80 + np.random.exponential(scale=280, size=num_photons)
    dt_vals = np.minimum(dt_vals, 2500)

    # Create a dictionary for the simulated photon data.
    photons = {
        'x': x_coords,
        'y': y_coords,
        'dt': dt_vals,
        'ms': ms_times
    }

    # --- Plotting ---

    # Plot 1: Scatter plot of photon positions over an 8x8 pixel area centered at (0,0)
    plt.figure(figsize=(6, 6))
    plt.scatter(photons['x'], photons['y'], alpha=0.7, label='Photon positions')
    plt.scatter([0], [0], color='red', marker='x', s=100, label='Fluorophore (0,0)')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Photon Positions in 8×8 Pixel Area')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: ms time trace (showing photon arrival times)
    plt.figure(figsize=(8, 4))
    plt.scatter(photons['ms'], np.ones_like(photons['ms']), marker='|', color='blue')
    plt.xlabel('Time (ms)')
    plt.yticks([])  # Hide y-axis ticks
    plt.title('Photon Arrival Times')
    plt.xlim(0, 400)
    plt.axvline(x=50, color='green', linestyle='--', label='Event Start (50 ms)')
    plt.axvline(x=350, color='red', linestyle='--', label='Event End (350 ms)')
    plt.legend()
    plt.show()