import numpy as np
import pandas as pd
import helper
from event import create_events
import fitting
import get_photons

def event_analysis(localizations_file, photons_file, drift_file, offset,
                   radius, int_time):
    """

    reads in file of localizations, connects events and analyzes them

    """
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')

    photons = helper.process_input(photons_file, dataset='photons')

    drift = helper.process_input(drift_file, dataset='drift')
    # first localizations to events
    events = create_events.locs_to_events(localizations, offset,
                                  box_side_length=radius,
                                  int_time=int_time)

    print('connected locs to events. total events: ', len(events))

    helper.validate_columns(events, ('event'))


    events_lt_avg_pos(events, photons, drift, offset, radius=radius,
                      int_time=int_time)
    helper.dataframe_to_picasso(
        events, localizations_file, '_event_new_avgroi')


def events_lt_avg_pos(event_file, photons_file,
                      drift_file, offset, radius=5,
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
    # drift = drift_file #pd.read_csv(drift_file, delimiter=' ',names =['x','y'])
    drift = helper.process_input(drift_file, dataset='drift')

    lifetime = np.ones(len(events))
    lt_photons = np.ones(len(events), dtype=int)
    x_position = np.ones(len(events))
    y_position = np.ones(len(events))
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
                                        box_side_length=radius,
                                        int_time=int_time)
        print('number of picked photons: ', len(pick_photons), '\n')
        print('picked area:   x - ', min(pick_photons['x']),
              max(pick_photons['x']))
        print('picked area:   y - ', min(pick_photons['y']),
              max(pick_photons['y']))

        print('__calibrate_peak__')
        peak_arrival_time = fitting.calibrate_peak(events_group, pick_photons,
                                           offset, box_side_length=radius,
                                           int_time=int_time)
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

            # print('my_event: \n', my_event)

            phot_event = pd.DataFrame(data=get_photons.crop_event
            (my_event, pick_photons, radius))

            if i == 0:
                print('FIRST fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
            elif i % 200 == 0:
                print('200 fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))

            x, y = fitting.avg_of_roi(my_event, phot_event, radius)

            x_position[i] = x
            y_position[i] = y
            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_event,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_event)
        counter += len(events_group)

    events['x'] = x_position
    events['y'] = y_position
    events['lifetime'] = lifetime
    events['lt_photons'] = lt_photons

    if isinstance(event_file, str):
        helper.dataframe_to_picasso(
            events, event_file, '_lt_avgPos_noBg')
    print('___________________FINISHED_____________________')
    print('\n', len(events), 'events tagged with lifetime and'
                       ' fitted with avg x,y position.')
