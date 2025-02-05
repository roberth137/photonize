import numpy as np
import pandas as pd
import helper
from event import create_events
import fitting
import get_photons
import time


def event_analysis(localizations_file, photons_file, drift_file, offset,
                   diameter, int_time, suffix='', max_dark_frames=1, proximity=2, filter_single=True):
    """

    reads in file of localizations, connects events and analyzes them

    """
    print('Starting event analysis: ...')
    start_read_locs = time.time()
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')
    end_read_locs = time.time()
    print(f'time to read in locs: {end_read_locs-start_read_locs}')
    # first localizations to events
    start_create_events = time.time()
    events = create_events.locs_to_events(localizations,
                                          offset=offset,
                                          int_time=int_time,
                                          max_dark_frames=max_dark_frames,
                                          proximity=proximity,
                                          filter_single=filter_single)
    end_create_events = time.time()
    print(f'time to create events: {end_create_events-start_create_events}')
    # read in photons and drift
    start_read_photons = time.time()
    photons = helper.process_input(photons_file, dataset='photons')
    end_read_photons = time.time()
    print(f'time to read photons: {end_read_photons-start_read_photons}')
    start_read_drift = time.time()
    drift = helper.process_input(drift_file, dataset='drift')
    end_read_drift = time.time()
    print(f'time to read drift: {end_read_drift-start_read_drift}')
    events_lt_avg_pos(events, photons, drift, offset, diameter=diameter,
                      int_time=int_time)
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
                                           position_fitting='averge_roi')
    helper.dataframe_to_picasso(
        events, localizations_file, file_extension, message)



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
    start_i_o = time.time()
    events = helper.process_input(event_file, dataset='locs')
    total_events = len(events)
    photons = helper.process_input(photons_file, dataset='photons')
    print(f'starting events_lt_avg_pos... ')
    print(len(photons), ' photons and ', total_events,
          'events read in')
    drift = helper.process_input(drift_file, dataset='drift')
    end_i_o = time.time()
    print(f'time for i/o: {end_i_o-start_i_o}')

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
    bg = np.ones(total_events, dtype=np.float32)
    bg_over_on = np.ones(total_events, dtype=np.float32)
    delta_x = np.ones(total_events, dtype=np.float32)
    delta_y = np.ones(total_events, dtype=np.float32)
    start_arr_time = time.time()
    peak_arrival_time = fitting.calibrate_peak_events(photons[:500000])
    start_dt = peak_arrival_time-0
    end_arr_time = time.time()
    print(f'arrival time calc: {end_arr_time-start_arr_time}.')

    print('peak arrival time:   ', peak_arrival_time)
    print('start time:          ', start_dt)

    counter = 0
    groups = set(events['group'])
    # iterating over every pick in file
    for g in groups:
        print('_______________________________________________________')
        print(f'Analysing group {int(g+1)} of {len(groups)}')
        start_pick = time.time()
        events_group = events[(events.group == g)]

        pick_photons = get_photons.get_pick_photons(
            events_group, photons, drift, offset,
            box_side_length=diameter, int_time=int_time)

        print('number of picked photons: ', len(pick_photons))
        print(pick_photons.head())

        end_pick = time.time()
        print(f'picking and undrifting time: {end_pick-start_pick}')
        #all_events_photons = all_events_photons[(all_events_photons.dt<1700)]
        #print(f'only considering events with dt < 1700')

        # iterating over every event in pick
        start_loop = time.time()
        for i in range(counter, counter + len(events_group)):
            if (i - counter) == 0:
                print('fitting lifetime of ', len(events_group),
                      ' events:')

            my_event = events.iloc[i]

            cylinder_photons = get_photons.crop_event(my_event,
                                                      pick_photons,
                                                      diameter, 0)

            start_ms , end_ms, duration_ms = fitting.get_on_off_dur(cylinder_photons,
                                                                    10,
                                                                    5)

            #filter photons according to new bounds
            num_photons_300_bg = len(cylinder_photons[(cylinder_photons.ms > (start_ms-150))
                                              &(cylinder_photons.ms < start_ms)
                                              |(cylinder_photons.ms < (end_ms+150))
                                              &(cylinder_photons.ms > end_ms)])

            photons_new_bounds = cylinder_photons[(cylinder_photons.ms >= start_ms)
                                                  &(cylinder_photons.ms <= end_ms)]

            #bg_total = my_event.bg * (diameter / 2) * np.pi
            total_photons = len(photons_new_bounds)
            phot_event = pd.DataFrame(data=photons_new_bounds)
            if i == 0:
                print('FIRST fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
            elif i % 200 == 0:
                print('200 fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
            x_arr = phot_event['x'].to_numpy()
            y_arr = phot_event['y'].to_numpy()
            x_t, y_t, sd_x_bg, sd_y_bg = fitting.event_position(x_arr,
                                                                y_arr,
                                                                return_sd=True)

            #calculate photon distances from new center for better lifetime determination:
            phot_x = np.copy(phot_event.x)
            phot_y = np.copy(phot_event.y)
            phot_x -= x_t
            phot_y -= y_t
            dist_2 = (phot_x**2 + phot_y**2)
            phot_event['distance'] = dist_2

            arrival_times = phot_event['dt'].to_numpy()
            distance_sq = phot_event['distance'].to_numpy()

            lifetime[i] = fitting.avg_lifetime_weighted_40(arrival_times,
                                                           distance_sq,
                                                           start_dt,
                                                           diameter)

            x_position[i] = x_t
            y_position[i] = y_t
            sdx[i] = sd_x_bg
            sdy[i] = sd_y_bg
            sdx_n[i] = sd_x_bg/np.sqrt(total_photons)
            sdy_n[i] = sd_y_bg/np.sqrt(total_photons)
            total_photons_lin[i] = total_photons
            start_ms_new[i] = start_ms
            end_ms_new[i] = end_ms
            duration_ms_new[i] = duration_ms
            brightness[i] = total_photons/duration_ms
            bg[i] = num_photons_300_bg/300
            bg_over_on[i] = len(cylinder_photons)/duration_ms
            delta_x[i] = my_event.x - x_t
            delta_y[i] = my_event.y - y_t
        end_loop = time.time()
        print(f'time for loooping over events: {end_loop - start_loop}.')
        counter += len(events_group)

    #events['x'] = x_position
    #events['y'] = y_position
    events['photons'] = total_photons_lin.astype(np.int32)
    events.insert(5, 'brightness', brightness)
    events.insert(6, 'lifetime', lifetime)
    events.insert(7, 'duration', duration_ms_new)
    events['bg'] = bg.astype(np.float32)
    events.insert(14, 'bg_over_on', bg_over_on.astype(np.float32))
    events.insert(15, 'sdx_n', sdx_n)
    events.insert(16, 'sdx', sdx.astype(np.float32))
    events.insert(17, 'sdy', sdy.astype(np.float32))
    #events['lpx'] = sdx_n.astype(np.float32)
    #events['lpy'] = sdy_n.astype(np.float32)
    events['start_ms'] = start_ms_new.astype(np.int32)
    events['end_ms'] = end_ms_new.astype(np.int32)
    events['delta_x'] = delta_x
    events['delta_y'] = delta_y
    events.drop(columns=['start_ms_fr', 'end_ms_fr'], inplace=True)

    if isinstance(event_file, str):
        helper.dataframe_to_picasso(
            events, event_file, 'eve_lt_avgPos')
    print('__________________________FINISHED____________________________')
    print(f'\n{len(events)} events tagged with lifetime and'
                       ' fitted with avg x,y position.')

    def events_lt_avg_pos_old(event_file, photons_file, drift_file, offset, diameter=5, int_time=200):
        # Read input files
        events = helper.process_input(event_file, dataset='locs')
        photons = helper.process_input(photons_file, dataset='photons')
        drift = helper.process_input(drift_file, dataset='drift')

        total_events = len(events)
        print(f'Starting events_lt_avg_pos with {len(photons)} photons and {total_events} events.')

        # Preallocate arrays for results
        result_arrays = {name: np.ones(total_events, dtype=np.float32) for name in [
            'lifetime', 'total_photons_lin', 'x_position', 'y_position',
            'sdx', 'sdy', 'sdx_n', 'sdy_n', 'duration_ms_new',
            'start_ms_new', 'end_ms_new', 'brightness', 'bg', 'bg_over_on',
            'delta_x', 'delta_y'
        ]}

        peak_arrival_time = fitting.calibrate_peak_events(photons[:500000])
        start_dt = peak_arrival_time
        print(f'Peak arrival time: {peak_arrival_time}, Start time: {start_dt}')

        # Grouped analysis of events
        for g in set(events['group']):
            print(f'Analyzing group {int(g + 1)} of {len(set(events["group"]))}')
            events_group = events[events.group == g]
            pick_photons = get_photons.get_pick_photons(events_group, photons, drift, offset, box_side_length=diameter,
                                                        int_time=int_time)

            for i, event in enumerate(events_group.itertuples(index=False)):
                cylinder_photons = get_photons.crop_event(event, pick_photons, diameter, 200)
                start_ms, end_ms, duration_ms = fitting.get_on_off_dur(cylinder_photons, 10, 5)

                # Filter photons and compute statistics
                photons_new_bounds = cylinder_photons[(cylinder_photons.ms > start_ms) & (cylinder_photons.ms < end_ms)]
                total_photons = len(photons_new_bounds)
                x_arr = photons_new_bounds['x'].to_numpy()
                y_arr = photons_new_bounds['y'].to_numpy()

                x_t, y_t, sd_x_bg, sd_y_bg = fitting.event_position(x_arr, y_arr, return_sd=True)
                phot_x = photons_new_bounds['x'].to_numpy() - x_t
                phot_y = photons_new_bounds['y'].to_numpy() - y_t
                dist_2 = phot_x ** 2 + phot_y ** 2

                arrival_times = photons_new_bounds['dt'].to_numpy()
                result_arrays['lifetime'][i] = fitting.avg_lifetime_weighted_40(arrival_times, dist_2, start_dt,
                                                                                diameter)

                # Store results
                result_arrays['x_position'][i] = x_t
                result_arrays['y_position'][i] = y_t
                result_arrays['sdx'][i] = sd_x_bg
                result_arrays['sdy'][i] = sd_y_bg
                result_arrays['sdx_n'][i] = sd_x_bg / np.sqrt(total_photons)
                result_arrays['sdy_n'][i] = sd_y_bg / np.sqrt(total_photons)
                result_arrays['total_photons_lin'][i] = total_photons
                result_arrays['start_ms_new'][i] = start_ms
                result_arrays['end_ms_new'][i] = end_ms
                result_arrays['duration_ms_new'][i] = duration_ms
                result_arrays['brightness'][i] = total_photons / duration_ms
                result_arrays['delta_x'][i] = event.x - x_t
                result_arrays['delta_y'][i] = event.y - y_t

        # Update events DataFrame
        for key, array in result_arrays.items():
            events[key] = array

        # Drop unnecessary columns and save results
        events.drop(columns=['start_ms_fr', 'end_ms_fr'], inplace=True)
        if isinstance(event_file, str):
            helper.dataframe_to_picasso(events, event_file, 'eve_lt_avgPos')

        print(f'Finished processing {len(events)} events.')
        return events

