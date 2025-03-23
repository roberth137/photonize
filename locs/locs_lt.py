import numpy as np
import pandas as pd
from utilities import helper
import fitting
import get_photons
import event


def locs_eve_lt_com_40(localizations_file, photons_file,
                    drift_file, offset, box_side_length=5,
                    int_time=200):
    """
    tagging list of picked localizations with lifetime
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
    localizations = pd.DataFrame(pd.read_hdf(localizations_file, key='locs'))
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = helper.process_input(drift_file, dataset='drift')

    events = event.locs_to_events(localizations, offset,
                                          box_side_length=box_side_length,
                                          int_time=int_time, filter_single=False)
    total_events = len(events)

    events_lifetime = np.ones(total_events, dtype=np.float32)
    lifetime = np.ones(total_localizations)
    lt_photons = np.ones(total_localizations, dtype=int)
    x_position = np.ones(total_localizations, dtype=np.float32)
    y_position = np.ones(total_localizations, dtype=np.float32)
    s_dev_x = np.ones(total_localizations, dtype=np.float32)
    s_dev_y = np.ones(total_localizations, dtype=np.float32)
    com_px = np.ones(total_localizations, dtype=np.float32)
    com_py = np.ones(total_localizations, dtype=np.float32)

    counter_locs = 0
    counter_events = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        events_group = events[(events.group == g)]
        pick_photons = get_photons.get_pick_photons(events_group, photons,
                                                    drift, offset,
                                                    box_side_length=box_side_length,
                                                    int_time=int_time)

        print('__calibrate_peak__')
        peak_arrival_time = fitting.calibrate_peak_arrival(pick_photons)
        print('peak arrival time is: ', peak_arrival_time, '_________________')

        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        for eve in range(counter_events, counter_events + len(events_group)):
            if (eve-counter_events) == 0: print('fitting lifetime of ', len(events_group),
                             ' events in this group.')
            this_event = events_group.iloc[eve - counter_events]
            this_event_photons = get_photons.crop_event(this_event,
                                                        pick_photons,
                                                        box_side_length)
            if eve % 200 == 0: print('200 events fitted. Number of photons',
                                   ' in last fit: ', len(this_event_photons))
            events_lifetime[eve] = fitting.avg_lifetime(this_event_photons,
                                                       peak_arrival_time)

        # iterating over every localization in pick
        for i in range(counter_locs, counter_locs + len(locs_group)):
            if (i-counter_locs) == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations in this group.')

            one_loc = locs_group.iloc[i - counter_locs]
            cylinder_photons = get_photons.crop_loc(one_loc, pick_photons, offset,
                                                    box_side_length, int_time)
            phot_loc = pd.DataFrame(data=cylinder_photons)

            bg_total = one_loc.bg * (box_side_length/2) * np.pi # calculates bg photons
            signal_photons = len(phot_loc) - bg_total

            if i % 200 == 0: print('200 locs fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            x, y , sd_x, sd_y = fitting.avg_of_roi_cons_bg(one_loc, one_loc.bg,
                                                           phot_loc, box_side_length,
                                                           return_sd=True)

            x_position[i] = x
            y_position[i] = y
            s_dev_x[i] = sd_x
            s_dev_y[i] = sd_y

            com_px[i] = fitting.localization_precision(signal_photons, sd_x, one_loc.bg)
            com_py[i] = fitting.localization_precision(signal_photons, sd_y, one_loc.bg)


            lifetime[i] = fitting.avg_lifetime(phot_loc, peak_arrival_time)
            lt_photons[i] = len(phot_loc)

        counter_locs += len(locs_group)
        counter_events += len(events_group)

    events_lifetime_pds = pd.Series(events_lifetime, index=range(1,(len(events)+1)))

    localizations['x'] = x_position.astype('float32')
    localizations['y'] = y_position.astype('float32')
    localizations['sdx'] = s_dev_x.astype('float32')
    localizations['sdy'] = s_dev_y.astype('float32')
    localizations['com_px'] = com_px.astype('float32')
    localizations['com_py'] = com_py.astype('float32')
    localizations['lifetime'] = lifetime.astype('float32')
    localizations['lt_photons'] = lt_photons
    localizations['eve_lt'] = localizations['event'].map(events_lifetime_pds)
    helper.dataframe_to_picasso(
        localizations, localizations_file, '_lt_com')
    print(len(localizations), 'localizations tagged with lifetime from fitting whole events and'
                              ' fitted individually with avg x,y position.')




def locs_lt_avg_pos_40(localizations_file, photons_file,
                    drift_file, offset, box_side_length=5,
                    integration_time=200):
    """
    tagging list of picked localizations with lifetime
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
    localizations = pd.DataFrame(pd.read_hdf(localizations_file, key='locs'))
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = helper.process_input(drift_file, dataset='drift')
    helper.calculate_total_photons(localizations, box_side_length=box_side_length)



    lifetime = np.ones(total_localizations)
    lt_photons = np.ones(total_localizations, dtype=int)
    x_position = np.ones(total_localizations, dtype=np.float32)
    y_position = np.ones(total_localizations, dtype=np.float32)
    #my_x_position = np.ones(total_localizations, dtype=np.float32)
    #my_y_position = np.ones(total_localizations, dtype=np.float32)
    #my_bg_array = np.ones(total_localizations, dtype=np.float32)
    s_dev_x = np.ones(total_localizations, dtype=np.float32)
    s_dev_y = np.ones(total_localizations, dtype=np.float32)
    sdx_sqrtn = np.ones(total_localizations, dtype=np.float32)
    sdy_sqrtn = np.ones(total_localizations, dtype=np.float32)
    #my_s_dev_x = np.ones(total_localizations, dtype=np.float32)
    #my_s_dev_y = np.ones(total_localizations, dtype=np.float32)
    com_px = np.ones(total_localizations, dtype=np.float32)
    com_py = np.ones(total_localizations, dtype=np.float32)
    #my_com_px = np.ones(total_localizations, dtype=np.float32)
    #my_com_py = np.ones(total_localizations, dtype=np.float32)
    old_lpx = np.copy(localizations['lpx'])
    old_lpy = np.copy(localizations['lpy'])


    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        pick_photons = get_photons.get_pick_photons(locs_group, photons,
                                        drift, offset,
                                        box_side_length, integration_time)

        peak_arrival_time = fitting.calibrate_peak_arrival(pick_photons)
        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            one_loc = locs_group.iloc[i - counter]
            cylinder_photons = get_photons.crop_loc(one_loc, pick_photons, offset,
                                                    box_side_length, integration_time)
            phot_loc = pd.DataFrame(data=cylinder_photons)
            total_photons = len(phot_loc)

            bg_total = one_loc.bg * (box_side_length/2) * np.pi # calculates bg photons
            #my_bg_total = my_bg * (diameter/2) * np.pi # calculates new bg photons
            signal_photons = len(phot_loc) - bg_total
            #signal_photons_new = len(phot_loc) - my_bg_total

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            x, y , sd_x, sd_y = fitting.avg_of_roi(one_loc, one_loc.bg,
                                                   phot_loc, box_side_length,
                                                   return_sd=True)

            x_position[i] = x
            y_position[i] = y

            s_dev_x[i] = sd_x
            s_dev_y[i] = sd_y
            sdx_sqrtn[i] = sd_x / np.sqrt(total_photons)
            sdy_sqrtn[i] = sd_y / np.sqrt(total_photons)

            #com_px[i] = fitting.localization_precision(signal_photons, sd_x, one_loc.bg)
            #com_py[i] = fitting.localization_precision(signal_photons, sd_y, one_loc.bg)


            lifetime[i] = fitting.avg_lifetime_weighted(phot_loc,
                                                       peak_arrival_time, box_side_length)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)

    localizations['x'] = x_position.astype('float32')
    localizations['y'] = y_position.astype('float32')
    localizations['lpx'] = sdx_sqrtn
    localizations['lpy'] = sdy_sqrtn
    localizations['sdx'] = s_dev_x.astype('float32')
    localizations['sdy'] = s_dev_y.astype('float32')
    #localizations['my_sdx'] = my_s_dev_x.astype('float32')
    #localizations['my_sdy'] = my_s_dev_y.astype('float32')
    localizations['com_px'] = com_px.astype('float32')
    localizations['com_py'] = com_py.astype('float32')
    localizations['old_lpx'] = old_lpx
    localizations['old_lpy'] = old_lpy
    #localizations['my_com_px'] = my_com_px.astype('float32')
    #localizations['my_com_py'] = my_com_py.astype('float32')
    localizations['lifetime'] = lifetime.astype('float32')
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file, '_lt_com')
    print(len(localizations), 'localizations tagged with lifetime and'
                              ' fitted with avg x,y position.')


def locs_lt_40(localizations_file, photons_file,
                    drift_file, offset, box_side_length=5,
                    integration_time=200):
    """
    tagging list of picked localizations with lifetime
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
    localizations = pd.DataFrame(pd.read_hdf(localizations_file, key='locs'))
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = helper.process_input(drift_file, dataset='drift')
    helper.calculate_total_photons(localizations, box_side_length=box_side_length)

    lifetime = np.ones(total_localizations)
    lt_photons = np.ones(total_localizations, dtype=int)

    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        pick_photons = get_photons.get_pick_photons(locs_group, photons,
                                        drift, offset,
                                        box_side_length, integration_time)

        peak_arrival_time = fitting.calibrate_peak_arrival(pick_photons)
        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            one_loc = locs_group.iloc[i - counter]
            cylinder_photons = get_photons.crop_loc(one_loc, pick_photons, offset,
                                                    box_side_length, integration_time)
            phot_loc = pd.DataFrame(data=cylinder_photons)

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))




            lifetime[i] = fitting.avg_lifetime_weighted(phot_loc,
                                                       peak_arrival_time, box_side_length)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)

    localizations['lifetime'] = lifetime.astype('float32')
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file, '_lt')
    print(len(localizations), 'localizations tagged with lifetime.')


def locs_lt_to_picasso_40(localizations_file, photons_file,
                          drift_file, offset, box_side_length=5,
                          integration_time=200):
    '''
    tagging list of picked localizations with lifetime (40mhz pulsed)
    and returning as picasso files
    IN:
    - list of picked localizations (picasso hdf5 file with 'group' column)
    - list of photons (hdf5 file)
    - drift (txt file)
    - offset (how many offsetted frames)
    OUT:
    - picasso hdf5 file tagged with lifetime
    - yaml file
    '''

    # read in files
    localizations = pd.read_hdf(localizations_file, key='locs')
    #assert isinstance(localizations, pd.DataFrame)
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    assert isinstance(photons, pd.DataFrame)
    drift = helper.process_input(drift_file, dataset='drift')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')


    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations), dtype=int)
    counter = 0

    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        pick_photons = get_photons.crop_undrift_crop(
            locs_group, photons,
            drift, offset,
            box_side_length, integration_time)

        peak_arrival_time = fitting.calibrate_peak_arrival(pick_photons)

        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):

            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            phot_loc = get_photons.crop_loc(
                locs_group.iloc[i - counter],
                pick_photons, offset,
                box_side_length, integration_time)

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            lifetime[i] = fitting.avg_lifetime(phot_loc,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_loc)

        counter += len(locs_group)


    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
