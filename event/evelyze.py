import numpy as np
import pandas as pd
from utilities import helper
from event import create_events
import fitting
import get_photons
from fitting import analyze_event
from typing import Optional, Dict, Tuple, Any

def event_analysis(localizations_file, photons_file, drift_file, offset,
                   diameter, int_time, suffix='', max_dark_frames=1,
                   proximity=2, filter_single=True, norm_brightness=False,
                   dt_window=None, more_ms=0, **kwargs):
    """

    reads in file of localizations, connects events and analyzes them

    """
    print('Starting event analysis: ...')
    # 1) read in files
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')
    photons = helper.process_input(photons_file, dataset='photons')
    drift = helper.process_input(drift_file, dataset='drift')
    # 2) create preliminary events by linking localizations
    events = create_events.locs_to_events(localizations,
                                          offset=offset,
                                          int_time=int_time,
                                          max_dark_frames=max_dark_frames,
                                          proximity=proximity,
                                          filter_single=filter_single)
    # 3) analyze events in main loop (localization+lifetime+brightness)
    arrival_time = {}
    events = events_lt_avg_pos(events, photons, drift,
                               offset, diameter=diameter,
                               int_time=int_time, arrival_time=arrival_time,
                               dt_window=dt_window, more_ms=more_ms, **kwargs)
    # 4) normalize brightness if applicable
    if norm_brightness:
        print('Normalizing brightness...')
        events = fitting.normalize_brightness(events)
    # 5) save events
    file_extension = '_event'+suffix
    message = helper.create_append_message(function='Evelyze',
                                           localizations_file=localizations_file,
                                           photons_file=photons_file,
                                           drift_file=drift_file,
                                           offset=offset,
                                           diameter=diameter,
                                           int_time=int_time,
                                           link_proximity=proximity,
                                           max_dark_frames=max_dark_frames,
                                           filter_single=filter_single,
                                           start_stop_event='ruptures-static',
                                           background='150ms-static',
                                           lifetime_fitting='quadratic_weight-static',
                                           position_fitting='averge_roi',
                                           peak_arrival_time=arrival_time['start'])
    helper.dataframe_to_picasso(
        events, localizations_file, file_extension, message)


def events_lt_avg_pos(event_file: str,
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
        print(f"_____Analyzing group {int(group_value + 1)} of {num_groups}_____")
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
            result = analyze_event(cylinder_photons, peak_arrival_time, diameter)

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
                print(f"Processed {idx} events; current event photon count: {result.num_photons}")
            idx += 1

    # Update events DataFrame with computed values
    sx_arr = events['sx'].to_numpy()
    sy_arr = events['sy'].to_numpy()
    bg_picasso = events['bg'].to_numpy()

    # Adjust total photon count using background correction
    photons_arr = total_photons_arr - (bg_picasso * duration_ms_arr / 200 * fit_area)

    events['x'] = x_position
    events['y'] = y_position
    events['photons'] = photons_arr.astype(np.float32)
    events['duration_ms'] = duration_ms_arr.astype(np.float32)
    events['lifetime_10ps'] = lifetime.astype(np.float32)
    events['brightness_phot_ms'] = (photons_arr / duration_ms_arr).astype(np.float32)
    events['bg'] = bg_picasso# * duration_ms_arr / 200
    events['lpx'] = fitting.localization_precision(sigma=sx_arr, photons=photons_arr, bg=bg_picasso, pixel_nm=7)
    events['lpy'] = fitting.localization_precision(sigma=sy_arr, photons=photons_arr, bg=bg_picasso, pixel_nm=7)
    #events['bg_picasso'] = bg_picasso.astype(np.float32)
    #events['brightness_phot_ms'] = (photons_arr / duration_ms_arr).astype(np.float32)
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





def events_lt_avg_pos_old(event_file, photons_file,
                      drift_file, offset, diameter=5,
                      int_time=200, arrival_time={},
                      dt_window=None, more_ms=0, **kwargs):
    """
    tagging list of events with lifetime and avg of roi position
    and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    - offset (how many offsetted frames)
    OUT:
    - picasso hdf5 file tagged with lifetime
    - yaml file
    """
    # read in files
    events = helper.process_input(event_file, dataset='locs')
    photons = helper.process_input(photons_file, dataset='photons')
    drift = helper.process_input(drift_file, dataset='drift')
    total_events = len(events)
    print(f'starting events_lt_avg_pos... \n')
    print(f'{len(photons)} photons and {total_events} events read in')

    #define arrays to store data
    lifetime = np.ones(total_events, dtype=np.float32)
    total_photons_arr = np.ones(total_events, dtype=np.float32)
    x_position = np.ones(total_events, dtype=np.float32)
    y_position = np.ones(total_events, dtype=np.float32)
    sdx = np.ones(total_events, dtype=np.float32)
    sdy = np.ones(total_events, dtype=np.float32)
    duration_ms_arr = np.ones(total_events, dtype=np.float32)
    start_ms_new = np.ones(total_events, dtype=np.float32)
    end_ms_new = np.ones(total_events, dtype=np.float32)
    delta_x = np.ones(total_events, dtype=np.float32)
    delta_y = np.ones(total_events, dtype=np.float32)
    bg_200ms_pixel = np.ones(total_events, dtype=np.float32)


    peak_arrival_time = fitting.calibrate_peak_arrival(photons[:500000])
    arrival_time['start'] = peak_arrival_time
    print('peak arrival time:   ', peak_arrival_time)
    if dt_window: print(f'considering photons with dt in {dt_window}')

    fit_area = (diameter/2)**2 * np.pi
    counter = 0
    groups = set(events['group'])
    # iterating over every pick in file
    for g in groups:
        print(f'__________Analysing group {int(g+1)} of {len(groups)}____________')
        events_group = events[(events.group == g)]
        pick_photons = get_photons.get_pick_photons(
            events_group, photons, drift, offset,
            diameter=diameter, int_time=int_time)
        print('number of picked photons: ', len(pick_photons))
        if dt_window:
            pick_photons = pick_photons[(pick_photons.dt>dt_window[0])&
                                        (pick_photons.dt<dt_window[1])]

        # iterating over every event in pick
        for i in range(counter, counter + len(events_group)):
            my_event = events.iloc[i]

            cylinder_photons = get_photons.crop_event(my_event,
                                                      pick_photons,
                                                      diameter,
                                                      more_ms=more_ms)

            result = analyze_event(cylinder_photons, peak_arrival_time, diameter)
            x_position[i] = result.x_fit
            y_position[i] = result.y_fit
            lifetime[i] = result.lifetime
            #sdx[i] = sd_x
            #sdy[i] = sd_y
            total_photons_arr[i] = result.num_photons
            start_ms_new[i] = result.start_ms
            end_ms_new[i] = result.end_ms
            duration_ms_arr[i] = result.duration_ms
            #bg_200ms_pixel[i] = num_bg_photons*(200/bg_measure_time)/fit_area
            #bg_over_on[i] = len(cylinder_photons)/duration_ms
            delta_x[i] = my_event.x - result.x_fit
            delta_y[i] = my_event.y - result.y_fit
            # console printing
            if (i - counter) == 0:
                print('fitting lifetime of ', len(events_group),
                      ' events:')
            if i == 0:
                print('FIRST fitted. Number of photons',
                      ' in phot_event: ', result.num_photons)
            elif i % 200 == 0:
                print('200 fitted. Number of photons',
                      ' in phot_event: ', result.num_photons)
        counter += len(events_group)

    sx_arr = np.copy(events.sx)
    sy_arr = np.copy(events.sy)
    bg_picasso = np.copy(events.bg)

    photons_arr = total_photons_arr - (bg_picasso * duration_ms_arr/200 * fit_area)

    events['x'] = x_position
    events['y'] = y_position
    events['photons'] = photons_arr.astype(np.float32)
    events['duration_ms'] = duration_ms_arr.astype(np.float32)
    events['lifetime_10ps'] = lifetime.astype(np.float32)
    events['bg'] = bg_picasso*duration_ms_arr/200
    events['lpx'] = fitting.localization_precision(sigma=sx_arr, photons=photons_arr, bg=bg_picasso, pixel_nm=7)
    events['lpy'] = fitting.localization_precision(sigma=sy_arr, photons=photons_arr, bg=bg_picasso, pixel_nm=7)
    events['bg_picasso'] = bg_picasso.astype(np.float32)
    events['brightness_phot_ms'] = (photons_arr/duration_ms_arr).astype(np.float32)
    events['start_ms'] = start_ms_new.astype(np.int32)
    events['end_ms'] = end_ms_new.astype(np.int32)
    events['delta_x'] = delta_x.astype(np.float32)
    events['delta_y'] = delta_y.astype(np.float32)
    events.drop(columns=['start_ms_fr', 'end_ms_fr'], inplace=True)

    if isinstance(event_file, str):
        helper.dataframe_to_picasso(
            events, event_file, 'eve_lt_avgPos')
    print('__________________________FINISHED____________________________')
    print(f'\n{len(events)} events tagged with lifetime and'
                       ' fitted with avg x,y position.')
    return events
