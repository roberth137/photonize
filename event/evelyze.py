import numpy as np
import pandas as pd
import helper
from event import create_events
import fitting
import get_photons

def event_analysis(localizations_file, photons_file, drift_file, offset,
                   diameter, int_time):
    """

    reads in file of localizations, connects events and analyzes them

    """
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')

    photons = helper.process_input(photons_file, dataset='photons')

    drift = helper.process_input(drift_file, dataset='drift')
    # first localizations to events
    events = create_events.locs_to_events(localizations, offset,
                                  box_side_length=diameter,
                                  int_time=int_time)

    print('connected locs to events. total events: ', len(events))

    helper.validate_columns(events, ('event'))


    events_lt_avg_pos(events, photons, drift, offset, diameter=diameter,
                      int_time=int_time)
    helper.dataframe_to_picasso(
        events, localizations_file, '_event')


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

    print(len(photons), ' photons and ', total_events,
          'events read in')
    print('starting events_lt_avg_pos function.')
    drift = helper.process_input(drift_file, dataset='drift')

    x_old = np.copy(events['x'])
    y_old = np.copy(events['y'])
    lpx_old = np.copy(events['lpx'])
    lpy_old = np.copy(events['lpy'])

    lifetime = np.ones(total_events, dtype=np.float32)
    total_photons_lin = np.ones(total_events, dtype=np.float32)
    x_position = np.ones(total_events, dtype=np.float32)
    y_position = np.ones(total_events, dtype=np.float32)
    #s_dev_x = np.ones(total_events, dtype=np.float32)
    #s_dev_y = np.ones(total_events, dtype=np.float32)
    s_dev_x_w_bg = np.ones(total_events, dtype=np.float32)
    s_dev_y_w_bg = np.ones(total_events, dtype=np.float32)
    com_px = np.ones(total_events, dtype=np.float32)
    com_py = np.ones(total_events, dtype=np.float32)
    #sdx_sqrtn = np.ones(total_events, dtype=np.float32)
    #sdy_sqrtn = np.ones(total_events, dtype=np.float32)
    sdx_sqrtn_w_bg = np.ones(total_events, dtype=np.float32)
    sdy_sqrtn_w_bg = np.ones(total_events, dtype=np.float32)

    counter = 0

    # iterating over every pick in file
    for g in set(events['group']):
        print('\n____________NEW GROUP________________')
        print(set(events['group']), '\n')
        print('this is group: ', g)

        events_group = events[(events.group == g)]

        print('__get_pick_photons___')
        pick_photons = get_photons.get_pick_photons(events_group, photons,
                                        drift, offset,
                                        box_side_length=diameter,
                                        int_time=int_time)
        print('number of picked photons: ', len(pick_photons), '\n')
        print('picked area:   x - ', min(pick_photons['x']),
              max(pick_photons['x']))
        print('picked area:   y - ', min(pick_photons['y']),
              max(pick_photons['y']))

        all_events_photons = get_photons.photons_of_many_events(events_group,
                                                                pick_photons,
                                                                diameter)
        all_events_photons = all_events_photons[(all_events_photons.dt<1600)]

        print('__calibrate_peak__')
        peak_arrival_time = 100#fitting.calibrate_peak_events(all_events_photons)
        print('peak arrival time is: ', peak_arrival_time, '_________________')
        print('_______________________________________________________')

        # iterating over every event in pick
        for i in range(counter, counter + len(events_group)):
            if (i - counter) == 0:
                print('fitting lifetime of ', len(events_group),
                      ' events.')
                i_values = range(counter, counter + len(events_group))
                print('i counter in range: ', i_values, '\n')

            my_event = events.iloc[i]

            cylinder_photons = get_photons.crop_event(my_event, all_events_photons, diameter)

            #column_names = list(cylinder_photons.columns)
            #print(column_names)
            bg_total = my_event.bg * (diameter / 2) * np.pi
            total_photons = len(cylinder_photons)
            signal_photons = total_photons - bg_total
            phot_event = pd.DataFrame(data=cylinder_photons)

            if i == 0:
                print('FIRST fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
            elif i % 200 == 0:
                print('200 fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))

            #x, y, sd_x, sd_y = fitting.event_position(my_event,
            #                                          phot_event,
            #                                          diameter,
            #                                          return_sd=True)
            x_t, y_t, sd_x_bg, sd_y_bg = fitting.event_position_w_bg(my_event,
                                                                 phot_event,
                                                                 diameter,
                                                                 return_sd=True)

            x_position[i] = x_t
            y_position[i] = y_t
            #s_dev_x[i] = sd_x
            #s_dev_y[i] = sd_y
            s_dev_x_w_bg[i] = sd_x_bg
            s_dev_y_w_bg[i] = sd_y_bg
            #sdx_sqrtn[i] = sd_x/np.sqrt(signal_photons)
            #sdy_sqrtn[i] = sd_y/np.sqrt(signal_photons)
            sdx_sqrtn_w_bg[i] = sd_x_bg/np.sqrt(total_photons)
            sdy_sqrtn_w_bg[i] = sd_y_bg/np.sqrt(total_photons)
            com_px[i] = fitting.localization_precision(signal_photons, sd_x_bg, my_event.bg)
            com_py[i] = fitting.localization_precision(signal_photons, sd_y_bg, my_event.bg)
            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_event,
                                                       peak_arrival_time)
            total_photons_lin[i] = total_photons
        counter += len(events_group)

    events['x'] = x_position
    events['y'] = y_position
    #events['sdx'] = s_dev_x
    #events['sdy'] = s_dev_y
    events['lpx'] = sdx_sqrtn_w_bg
    events['lpy'] = sdy_sqrtn_w_bg
    events['sdx'] = s_dev_x_w_bg
    events['sdy'] = s_dev_y_w_bg
    events['old_lpx'] = lpx_old
    events['old_lpy'] = lpy_old
    #events['sdx_sqrtn'] = sdx_sqrtn
    #events['sdy_sqrtn'] = sdy_sqrtn
    events['com_px'] = com_px
    events['com_py'] = com_py
    events['lifetime'] = lifetime
    events['tot_phot_cylinder'] = total_photons_lin
    events['x_old'] = x_old
    events['y_old'] = y_old
    events['delta_x'] = (events['x_old'] - events['x'])
    #events['log_delta_x2_inverse'] = 1/(-np.log((events['x_old'] - events['x'])**2))

    if isinstance(event_file, str):
        helper.dataframe_to_picasso(
            events, event_file, 'eve_lt_avgPos')
    print('___________________FINISHED_____________________')
    print('\n', len(events), 'events tagged with lifetime and'
                       ' fitted with avg x,y position.')
