# This is the main function for analysing a single event

import numpy as np
import pandas as pd

from fitting.on_off import get_on_off_dur
from fitting.localization import localize_com
from fitting.localization import mle_fixed_sigma_bg
from fitting.lifetime import avg_lifetime_weighted
from collections import namedtuple
import matplotlib.pyplot as plt


# Define the named tuple once
EventResult = namedtuple('EventResult', [
    'x_fit', 'y_fit', 'lifetime',
    'start_ms', 'end_ms', 'duration_ms',
    'num_photons'
])

def fit_event(photons, x_start, y_start, sigma, bg_rate, dt_peak, diameter):
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

    # filter photons within the event time window and after pulse
    event_photons = photons[(photons.ms >= start_ms) & (photons.ms <= end_ms) & (photons.dt >= dt_peak)]

    # extract coordinates and arrival times
    x_photons = np.copy(event_photons.x)
    y_photons = np.copy(event_photons.y)
    dt_photons = np.copy(event_photons.dt)

    # fit x and y position using center of mass
    #x_fit, y_fit, sdx, sdy = localize_com(
    #    event_photons.x, event_photons.y, return_sd=True
    #)
    result = mle_fixed_sigma_bg(x_photons,
                                            y_photons,
                                            x_start=x_start,
                                            y_start=y_start,
                                            diameter=diameter,
                                            sigma=sigma,
                                            bg_rate=bg_rate,
                                            binding_time=duration_ms)
    x_fit = result['mu_x']
    y_fit = result['mu_y']

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


def simulate_event(seed=42,
                   bg_200ms_px=3,
                   dur_all=400,
                   dur_eve=300,
                   sx=1.1,
                   lifetime=350,
                   peak_dt=80,
                   brightness_eve=2,
                   size_px=8):
    """
    Simulates a time trace with background and event photons, returned as a sorted DataFrame.
    """
    np.random.seed(seed)

    total_area = size_px**2
    n_bg_photons_total = int(bg_200ms_px * total_area * (dur_all / 200))
    n_eve_total = int(dur_eve * brightness_eve)

    # 1. Background photons
    bg_ms = np.sort(np.random.uniform(0, dur_all, size=n_bg_photons_total))
    bg_x = np.random.uniform(low=0, high=size_px, size=n_bg_photons_total) - (size_px / 2)
    bg_y = np.random.uniform(low=0, high=size_px, size=n_bg_photons_total) - (size_px / 2)
    bg_dt = np.clip(np.random.exponential(scale=lifetime, size=n_bg_photons_total) + peak_dt, 0, 2500)

    # 2. Event photons
    event_start = (dur_all - dur_eve) / 2
    event_end = event_start + dur_eve

    event_ms = np.sort(np.random.uniform(event_start, event_end, size=n_eve_total))
    event_x = np.random.normal(loc=0, scale=sx, size=n_eve_total)
    event_y = np.random.normal(loc=0, scale=sx, size=n_eve_total)
    event_dt = np.clip(np.random.exponential(scale=lifetime, size=n_eve_total) + peak_dt, 0, 2500)

    # Combine into a DataFrame
    photons_bg = pd.DataFrame({
        'x': bg_x,
        'y': bg_y,
        'dt': bg_dt,
        'ms': bg_ms
    })
    photons_eve = pd.DataFrame({
        'x': event_x,
        'y': event_y,
        'dt': event_dt,
        'ms': event_ms
    })

    # Sort by time (ms)
    #photons = photons.sort_values(by='ms').reset_index(drop=True)

    return photons_eve, photons_bg, peak_dt

#test

if __name__ == "__main__":
    print('main loop')
    # 1. Generate ms time stamps using the simulation function
    photons_eve, photons_bg, peak_dt = simulate_event(
        seed=42,
        bg_200ms_px=3,
        dur_all=400,
        dur_eve=300,
        brightness_eve=2  # 2 photons per ms during the event
    )

    print(photons_eve)
    print('______________________________________________')
    print(photons_bg)
    diameter=4.5

    all_photons = pd.concat([photons_eve, photons_bg])
    all_photons['distance'] = np.sqrt((all_photons.x**2) + (all_photons.y**2))
    roi_photons = all_photons[all_photons['distance'] < diameter/2]

    result = fit_event(roi_photons, peak_dt, diameter=diameter)

    print(result)
    num_photons = len(roi_photons)


    # Plot 1: Scatter plot of photon positions over an 8x8 pixel area centered at (0,0)
    plt.figure(figsize=(6, 6))
    plt.scatter(roi_photons['x'], roi_photons['y'], alpha=0.7, label='Photon positions')
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
    plt.scatter(roi_photons['ms'], np.ones_like(roi_photons['ms']), marker='|', color='blue')
    plt.xlabel('Time (ms)')
    plt.yticks([])  # Hide y-axis ticks
    plt.title('Photon Arrival Times')
    plt.xlim(0, 400)
    plt.axvline(x=50, color='green', linestyle='--', label='Event Start (50 ms)')
    plt.axvline(x=350, color='red', linestyle='--', label='Event End (350 ms)')
    plt.legend()
    plt.show()