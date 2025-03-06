import numpy as np
import pandas as pd
import helper
from event import create_events
import fitting
import get_photons
import time
from fitting import normalize_brightness


def event_analysis(localizations_file, photons_file, drift_file, offset,
                   diameter, int_time, suffix='', max_dark_frames=1,
                   proximity=2, filter_single=True, norm_brightness=False, **kwargs):
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
    events = create_events.locs_to_events(localizations,
                                          offset=offset,
                                          int_time=int_time,
                                          max_dark_frames=max_dark_frames,
                                          proximity=proximity,
                                          filter_single=filter_single)
    # read in photons and drift
    photons = helper.process_input(photons_file, dataset='photons')
    drift = helper.process_input(drift_file, dataset='drift')
    arrival_time = {}
    events = events_lt_avg_pos(events, photons, drift, offset, diameter=diameter,
                      int_time=int_time, arrival_time=arrival_time, **kwargs)
    if norm_brightness:
        print('Normalizing brightness...')
        laser_profile = fitting.get_laser_profile(localizations[::offset])
        events = normalize_brightness(events, laser_profile)
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


def events_lt_avg_pos(event_file, photons_file,
                      drift_file, offset, diameter=5,
                      int_time=200, arrival_time={}, **kwargs):
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
    tailcut = kwargs.get('tailcut')

    lifetime = np.ones(total_events, dtype=np.float32)
    lplt = np.ones(total_events, dtype=np.float32)
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


    peak_arrival_time = fitting.calibrate_peak_events(photons[:500000])
    start_dt = peak_arrival_time-0
    arrival_time['start'] = start_dt
    lpx_arr = np.copy(events.lpx)
    lpy_arr = np.copy(events.lpy)

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
            diameter=diameter, int_time=int_time)

        print('number of picked photons: ', len(pick_photons))
        end_pick = time.time()
        print(f'picking and undrifting time: {end_pick-start_pick}')
        pick_photons = pick_photons[(pick_photons.dt<1680)]
        #print(f'only considering events with dt < 1700')

        # iterating over every event in pick
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
            new_eve = pd.DataFrame([{'start_ms': start_ms, 'end_ms': end_ms, 'x': my_event.x, 'y': my_event.y}])
            this_eve = new_eve.iloc[0]
            photons_new_bounds = get_photons.crop_event(this_eve, pick_photons, diameter)

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

            if tailcut is not None:
                phot_event = phot_event[(phot_event['dt']<tailcut)]
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
        counter += len(events_group)

    events['x'] = x_position
    events['y'] = y_position
    events['photons'] = total_photons_lin.astype(np.int32)
    events['lpx'] = sdx_n
    events['lpy'] = sdy_n
    events.insert(5, 'brightness_phot_ms', brightness)
    events.insert(6, 'lifetime_10ps', lifetime)
    events.insert(7, 'duration_ms', duration_ms_new)
    events.insert(8, 'lplt', (lifetime/total_photons_lin).astype(np.float32))
    events['bg'] = bg.astype(np.float32)
    events.insert(14, 'bg_over_on', bg_over_on.astype(np.float32))
    events.insert(15, 'old_lpx', lpx_arr)
    events.insert(16, 'old_lpy', lpy_arr)
    events.insert(17, 'sdx', sdx.astype(np.float32))
    events.insert(18, 'sdy', sdy.astype(np.float32))
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
    return events
