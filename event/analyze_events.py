import numpy as np
import pandas as pd
from utilities import helper
import fitting
import get_photons
from fitting import fit_event
from typing import Optional, Dict, Tuple, Any

def events_lt_pos(event_file: str,
                      photons_file: str,
                      drift_file: str,
                      offset: int,
                      diameter: float = 5,
                      int_time: int = 200,
                      arrival_time: Optional[Dict[str, Any]] = None,
                      dt_window: Optional[Tuple[float, float]] = None,
                      more_ms: int = 0,
                      **kwargs) -> pd.DataFrame:
    """
    Tag events with lifetime and average ROI position and return as picasso files.

    Parameters:
        event_file: Path to the event file (picasso hdf5 with 'group' column).
        photons_file: Path to the photons hdf5 file.
        drift_file: Path to the drift txt file.
        offset: Frame offset.
        diameter: Diameter for region of interest.
        int_time: Integration time.
        arrival_time: Dictionary to store arrival time information (will be initialized if None).
        dt_window: Optional time window to filter dt values.
        more_ms: Extra milliseconds for cropping.
        **kwargs: Additional keyword arguments.

    Returns:
        A pandas DataFrame of events tagged with lifetime and fitted with average x, y positions.
    """
    print("Starting events_lt_avg_pos...")

    # Ensure mutable default is not reused
    if arrival_time is None:
        arrival_time = {}

    # Read input files
    events = helper.process_input(event_file, dataset='locs')
    photons = helper.process_input(photons_file, dataset='photons')
    drift = helper.process_input(drift_file, dataset='drift')
    total_events = len(events)
    print(f"{len(photons)} photons and {total_events} events read in")

    # Preallocate arrays using np.empty (assumes all values will be assigned)
    lifetime = np.empty(total_events, dtype=np.float32)
    total_photons_arr = np.empty(total_events, dtype=np.float32)
    x_position = np.empty(total_events, dtype=np.float32)
    y_position = np.empty(total_events, dtype=np.float32)
    sdx = np.empty(total_events, dtype=np.float32)
    sdy = np.empty(total_events, dtype=np.float32)
    duration_ms_arr = np.empty(total_events, dtype=np.float32)
    start_ms_new = np.empty(total_events, dtype=np.float32)
    end_ms_new = np.empty(total_events, dtype=np.float32)
    delta_x = np.empty(total_events, dtype=np.float32)
    delta_y = np.empty(total_events, dtype=np.float32)

    # Calibrate peak arrival time from a subset of photons
    peak_arrival_time = fitting.calibrate_peak_arrival(photons[:500000])
    arrival_time['start'] = peak_arrival_time
    print(f"Peak arrival time: {peak_arrival_time}")
    if dt_window:
        print(f"Considering photons with dt in {dt_window}")

    # Calculate area for photon fitting (could be named constant)
    fit_area = (diameter / 2) ** 2 * np.pi

    # Initialize event index
    idx = 0

    # Iterate over events by group using groupby
    num_groups = events['group'].nunique()
    for group_value, events_group in events.groupby('group'):
        print(f"__________Analyzing group {int(group_value + 1)} of {num_groups}__________")
        print(f'{len(events_group)} events in current group.')
        pick_photons = get_photons.get_pick_photons(
            events_group, photons, drift, offset,
            diameter=diameter, int_time=int_time
        )
        print(f"Number of picked photons: {len(pick_photons)}")

        # Apply dt window filter if provided
        if dt_window:
            pick_photons = pick_photons[(pick_photons.dt > dt_window[0]) &
                                        (pick_photons.dt < dt_window[1])]

        # Iterate over each event in the current group
        for i, my_event in events_group.iterrows():
            # Crop the relevant photons for this event
            cylinder_photons = get_photons.crop_event(my_event, pick_photons, diameter, more_ms=more_ms)

            # Analyze the event using a helper function (assumed to return an object with attributes)
            result = fit_event(cylinder_photons, peak_arrival_time, diameter)

            # Store computed values
            x_position[idx] = result.x_fit
            y_position[idx] = result.y_fit
            lifetime[idx] = result.lifetime
            total_photons_arr[idx] = result.num_photons
            start_ms_new[idx] = result.start_ms
            end_ms_new[idx] = result.end_ms
            duration_ms_arr[idx] = result.duration_ms
            delta_x[idx] = my_event.x - result.x_fit
            delta_y[idx] = my_event.y - result.y_fit
            if idx % 200 == 0:
                print(f"Processed {idx} events; current event photons: {result.num_photons}")
            idx += 1

    # Update events DataFrame with computed values
    sx_arr = events['sx'].to_numpy()
    sy_arr = events['sy'].to_numpy()
    bg_picasso = events['bg'].to_numpy()

    # Adjust total photon count using background correction
    photons_arr = total_photons_arr - (bg_picasso * duration_ms_arr / 200 * fit_area)

    #frame
    #event
    events['x'] = x_position
    events['y'] = y_position
    events['photons'] = photons_arr.astype(np.float32)
    events.insert(5, 'duration_ms', duration_ms_arr.astype(np.float32))
    events.insert(6, 'lifetime_10ps', lifetime.astype(np.float32))
    events.insert(7, 'brightness_phot_ms', (photons_arr / duration_ms_arr).astype(np.float32))
    events['bg'] = bg_picasso# * duration_ms_arr / 200
    events['lpx'] = fitting.localization_precision(sigma=sx_arr, photons=photons_arr, bg=bg_picasso, pixel_nm=7)
    events['lpy'] = fitting.localization_precision(sigma=sy_arr, photons=photons_arr, bg=bg_picasso, pixel_nm=7)
    events['start_ms'] = start_ms_new.astype(np.int32)
    events['end_ms'] = end_ms_new.astype(np.int32)
    events['delta_x'] = delta_x.astype(np.float32)
    events['delta_y'] = delta_y.astype(np.float32)
    events.drop(columns=['start_ms_fr', 'end_ms_fr'], inplace=True)

    # Save to picasso file if event_file is provided as a string
    if isinstance(event_file, str):
        helper.dataframe_to_picasso(events, event_file, 'eve_lt_avgPos')

    print(f"_______________FINISHED: {len(events)} events analysed!_______________")
    return events