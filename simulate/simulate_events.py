import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import simulate as s

# Option A: Load event statistics from an HDF5 file
event_stats = pd.read_hdf('simulate/sim_experiments_stats/2green_conditions.hdf5')

# Option B: Or generate 10000 event statistics instead (uncomment if desired)
# event_stats = s.simulate_event_stats(n_events=10000)
# event_stats = pd.DataFrame(event_stats)  # Convert structured array to a DataFrame

print("First 5 event stats:")
print(event_stats.head(5))

# -------------------------------
# 2. Use event stats as parameters for simulation
# -------------------------------

n = len(event_stats)
x_fit_w_bg = np.empty(n, dtype=float)
y_fit_w_bg = np.empty(n, dtype=float)
x_fit_pure = np.empty(n, dtype=float)
y_fit_pure = np.empty(n, dtype=float)

# Iterate over DataFrame rows
for i, row in event_stats.iterrows():
    # Extract parameters for the event from the row
    # If 'photons' is float, we take int(...) for the photon count
    num_photons = int(row['photons'])

    # Use the average of sx and sy as the effective PSF width
    sigma_psf = (row['sx'] + row['sy']) / 2.0

    binding_time_ms = row['binding_time']
    bg_rate_true = row['bg']

    # Simulate fluorophore
    x_fluo, y_fluo = s.simulate_fluorophore(num_photons=num_photons,
                                            sigma_psf=sigma_psf,
                                            camera_error=s.camera_error,
                                            min_cam_binning=s.subpixel)

    # Simulate background
    x_bg, y_bg = s.simulate_background(num_pixels=s.num_pixels,
                                       binding_time_ms=binding_time_ms,
                                       bg_rate_true=bg_rate_true,
                                       subpixel=s.subpixel)

    # Perform COM fit without background correction
    pos_no_bg = s.analyze_sim_event(
        x_fluo, y_fluo,
        x_bg, y_bg,
        x_entry=s.x_ref, y_entry=s.y_ref,
        diameter=s.fitting_diameter,
        consider_bg=False
    )

    # Perform COM fit with background correction
    pos_with_bg = s.analyze_sim_event(
        x_fluo, y_fluo,
        x_bg, y_bg,
        x_entry=s.x_ref, y_entry=s.y_ref,
        diameter=s.fitting_diameter,
        consider_bg=True
    )

    # If no photons were found within the ROI, skip this event.
    if pos_no_bg[0] is None or pos_with_bg[0] is None:
        x_fit_pure[i], y_fit_pure[i] = np.nan, np.nan
        x_fit_w_bg[i], y_fit_w_bg[i] = np.nan, np.nan
        continue

    # Store fitted positions
    x_fit_pure[i], y_fit_pure[i] = pos_no_bg
    x_fit_w_bg[i], y_fit_w_bg[i] = pos_with_bg

# After the loop, remove NaNs if any were inserted for skipped events
valid_mask = ~np.isnan(x_fit_pure) & ~np.isnan(x_fit_w_bg)
x_fit_pure = x_fit_pure[valid_mask]
y_fit_pure = y_fit_pure[valid_mask]
x_fit_w_bg = x_fit_w_bg[valid_mask]
y_fit_w_bg = y_fit_w_bg[valid_mask]

# Calculate distances from (0, 0) – or use s.x_ref, s.y_ref if needed
_, _, distance_pure = s.distance_to_point(x_fit_pure, y_fit_pure, x_ref=0, y_ref=0)
_, _, distance_w_bg = s.distance_to_point(x_fit_w_bg, y_fit_w_bg, x_ref=0, y_ref=0)

# -------------------------------
# 3. Plot the fitted positions
# -------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(distance_w_bg, bins=30, range=(0, 0.3), color='purple', alpha=0.7)
plt.xlabel('Error distance (pixels)')
plt.ylabel('Counts')
plt.title(f'Error w/ BG correction (std: {np.std(distance_w_bg):.5f})')

plt.subplot(1, 2, 2)
plt.hist(distance_pure, bins=30, range=(0, 0.3), color='green', alpha=0.7)
plt.xlabel('Error distance (pixels)')
plt.ylabel('Counts')
plt.title(f'Error w/o BG correction (std: {np.std(distance_pure):.5f})')

plt.tight_layout()
plt.show()
