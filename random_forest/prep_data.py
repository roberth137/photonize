import pandas as pd
import numpy as np
import helper
import get_photons
import fitting

input_events = ''
input_photons = ''
drift_file = ''
fluorophore = 'Cy3'
offset = 10
diameter = 4.5
int_time = 200

events = helper.process_input(input_events, 'locs')
photons = helper.process_input(input_photons, 'photons')
drift = helper.process_input(drift_file, 'drift')

peak_arrival_time = fitting.calibrate_peak_events(photons[:1000000])
max_dt = max(photons[:1000000].dt)

# Parameters
bin_size = 5  # Bin size for histogramming (in the same units as dt)
bins = np.arange(peak_arrival_time, max_dt, bin_size)

output_file = "histogram_data.csv"


pick_photons = get_photons.get_pick_photons(events, photons,
                                        drift, offset,
                                        box_side_length=diameter,
                                        int_time=int_time)
# now photons are undrifted

histograms = pd.DataFrame()

for _, event in events.iterrows():
    event_photons = get_photons.crop_cylinder(event.x,
                                         event.y,
                                         event.s_ms_new,
                                         event.e_ms_new,
                                         diameter,
                                         pick_photons)

    hist, _ = np.histogram(event_photons.dt, bins=bins)

    hist = hist / hist.max() if hist.max() > 0 else hist

    hist_series = pd.Series(hist)

    pd.concat([histograms, hist_series], axis=0)


# 1. Read in events (only 1 group to start)

# 2. Undrift photons

# 3. Iterate events
    # For every event: Get photons
    # Histogram photons
    # Add counts to dataframe

# 4. Label data

# 5. Save file
'''
def generate_histograms(events_df, photons_df):
    histograms = []
    for _, event in events_df.iterrows():
        group = event["group"]

        # Select photons corresponding to the group
        group_photons = photons_df[photons_df["group"] == group]

        # Collect photon arrival times (dt)
        dt_values = group_photons["dt"].values

        # Generate histogram for dt_values
        bins = np.arange(peak_dt, max_dt + bin_size, bin_size)
        hist, _ = np.histogram(dt_values, bins=bins)

        # Normalize histogram to make the max value 1
        hist = hist / hist.max() if hist.max() > 0 else hist

        # Add label (cy3 or a56) and histogram values to the list
        label = event["label"]
        histograms.append([label] + hist.tolist())

    # Create a DataFrame for the histograms
    histogram_columns = ["label"] + [f"bin_{i}" for i in range(len(bins) - 1)]
    histogram_df = pd.DataFrame(histograms, columns=histogram_columns)

    return histogram_df


# Example: Reading DataFrames (replace with actual file paths or data sources)
events_df = pd.read_csv("events.csv")  # Contains x, y, duration, group, label
photons_df = pd.read_csv("photons.csv")  # Contains x, y, dt, group
'''