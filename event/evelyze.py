import numpy as np
import pandas as pd
import helper
from event import create_events
import fitting
import get_photons
import ruptures as rpt

def event_analysis(localizations_file, photons_file, drift_file, offset,
                   diameter, int_time, suffix=''):
    """

    reads in file of localizations, connects events and analyzes them

    """
    print('Starting event analysis: ...')
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')
    photons = helper.process_input(photons_file, dataset='photons')
    drift = helper.process_input(drift_file, dataset='drift')

    # first localizations to events
    events = create_events.locs_to_events(localizations, offset,
                                  box_side_length=diameter,
                                  int_time=int_time, filter_single=True)

    events_lt_avg_pos(events, photons, drift, offset, diameter=diameter,
                      int_time=int_time)
    file_extension = '_event'+suffix
    helper.dataframe_to_picasso(
        events, localizations_file, file_extension)


def events_lt_avg_pos(event_file, photons_file,
                      drift_file, offset, diameter=5,
                      int_time=200):
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
    total_events = len(events)
    photons = helper.process_input(photons_file, dataset='photons')
    print(f'starting events_lt_avg_pos... ')
    print(len(photons), ' photons and ', total_events,
          'events read in')
    drift = helper.process_input(drift_file, dataset='drift')

    lifetime = np.ones(total_events, dtype=np.float32)
    total_photons_lin = np.ones(total_events, dtype=np.float32)
    x_position = np.ones(total_events, dtype=np.float32)
    y_position = np.ones(total_events, dtype=np.float32)
    sdx = np.ones(total_events, dtype=np.float32)
    sdy = np.ones(total_events, dtype=np.float32)
    sdx_n = np.ones(total_events, dtype=np.float32)
    sdy_n = np.ones(total_events, dtype=np.float32)
    duration_ms_new = np.ones(total_events, dtype=np.float32)
    start_ms_new = np.ones(total_events, dtype=np.float32)
    end_ms_new = np.ones(total_events, dtype=np.float32)
    brightness = np.ones(total_events, dtype=np.float32)

    peak_arrival_time = fitting.calibrate_peak_events(photons[:500000])
    start_dt = peak_arrival_time-0
    print('peak arrival time:   ', peak_arrival_time)
    print('start time:          ', start_dt)

    counter = 0
    groups = set(events['group'])
    # iterating over every pick in file
    for g in groups:
        print('_______________________________________________________________')
        print(f'Analysing group {int(g+1)} of {len(groups)}')

        events_group = events[(events.group == g)]

        pick_photons = get_photons.get_pick_photons(events_group, photons,
                                        drift, offset,
                                        box_side_length=diameter,
                                        int_time=int_time)
        print('number of picked photons: ', len(pick_photons))

        all_events_photons = get_photons.photons_of_many_events(events_group,
                                                                pick_photons,
                                                                diameter, 300)
        print('number of event photons: ', len(all_events_photons))

        #all_events_photons = all_events_photons[(all_events_photons.dt<1700)]
        #print(f'only considering events with dt < 1700')

        # iterating over every event in pick
        for i in range(counter, counter + len(events_group)):
            if (i - counter) == 0:
                print('fitting lifetime of ', len(events_group),
                      ' events:')

            my_event = events.iloc[i]

            cylinder_photons = get_photons.crop_event(my_event, all_events_photons, diameter, 300)

            #determine start and end of event
            bin_size = 10
            bins = np.arange(min(cylinder_photons.ms), max(cylinder_photons.ms) + bin_size, bin_size)
            counts, _ = np.histogram(cylinder_photons, bins=bins)
            smoothed_counts_1 = lee_filter_1d(counts, 5)
            model = "l2"  # Least squares cost function
            algo = rpt.Binseg(model=model, min_size=1, jump=1).fit(smoothed_counts_1)
            change_points = algo.predict(n_bkps=2)  # Detect 2 change points (for on and off)
            change_points_trans = np.array(change_points)
            ms_dur = (change_points_trans[1]-change_points_trans[0])*bin_size
            change_points_trans[0] = (change_points_trans[0] - 1.5) * bin_size + bins[0]
            change_points_trans[1] = (change_points_trans[1] + 0.5) * bin_size + bins[0]

            #filter photons according to new bounds
            photons_new_bounds = cylinder_photons[(cylinder_photons.ms > change_points_trans[0])
            &(cylinder_photons.ms < change_points_trans[1])]

            bg_total = my_event.bg * (diameter / 2) * np.pi
            total_photons = len(cylinder_photons)
            signal_photons = total_photons - bg_total
            phot_event = pd.DataFrame(data=photons_new_bounds)
            if i == 0:
                print('FIRST fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
            elif i % 200 == 0:
                print('200 fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))

            x_t, y_t, sd_x_bg, sd_y_bg = fitting.event_position(my_event,
                                                                phot_event,
                                                                diameter,
                                                                return_sd=True)

            #calculate photon distances from new center for better lifetime determination:
            phot_x = np.copy(phot_event.x)
            phot_y = np.copy(phot_event.y)
            phot_x -= x_t
            phot_y -= y_t
            dist_2 = (phot_x**2 + phot_y**2)
            phot_event['distance'] = dist_2

            lifetime[i] = fitting.avg_lifetime_weighted_40(phot_event,
                                                          start_dt,
                                                          diameter)

            x_position[i] = x_t
            y_position[i] = y_t
            sdx[i] = sd_x_bg
            sdy[i] = sd_y_bg
            sdx_n[i] = sd_x_bg/np.sqrt(total_photons)
            sdy_n[i] = sd_y_bg/np.sqrt(total_photons)
            total_photons_lin[i] = total_photons
            start_ms_new[i] = change_points_trans[0]
            end_ms_new[i] = change_points_trans[1]
            duration_ms_new[i] = ms_dur
            brightness[i] = total_photons/ms_dur

        counter += len(events_group)

    events['x'] = x_position
    events['y'] = y_position
    events['photons'] = total_photons_lin
    events.insert(5, 'brightness', brightness)
    events.insert(6, 'lifetime', lifetime)
    events.insert(7, 'duration', duration_ms_new)
    events['lpx'] = sdx_n
    events['lpy'] = sdy_n
    events['sdx'] = sdx
    events['sdy'] = sdy
    events['start_ms'] = start_ms_new
    events['end_ms'] = end_ms_new
    events.drop(columns=['start_ms_fr', 'end_ms_fr'], inplace=True)

    if isinstance(event_file, str):
        helper.dataframe_to_picasso(
            events, event_file, 'eve_lt_avgPos')
    print('___________________FINISHED_____________________')
    print(f'\n{len(events)} events tagged with lifetime and'
                       ' fitted with avg x,y position.')




def lee_filter_1d(data, window_size=5):
    """
    Applies the Lee filter to 1D data for noise reduction.

    Parameters:
        data (numpy.ndarray): 1D array of data to filter.
        window_size (int): Size of the sliding window (must be odd).

    Returns:
        numpy.ndarray: Smoothed data after applying the Lee filter.
    """
    # Ensure the window size is odd
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")

    # Calculate the local mean and variance in the sliding window
    padded_data = np.pad(data, pad_width=window_size // 2, mode='reflect')
    local_mean = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    local_var = np.convolve(padded_data ** 2, np.ones(window_size) / window_size, mode='valid') - local_mean ** 2

    # Estimate the noise variance (assume it's uniform across the data)
    noise_var = np.mean(local_var)

    # Apply the Lee filter
    result = local_mean + (local_var / (local_var + noise_var)) * (data - local_mean)
    return result