import numpy as np
import pandas as pd
import helper
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

    lifetime = np.ones(total_localizations)
    lt_photons = np.ones(total_localizations, dtype=int)
    x_position = np.ones(total_localizations, dtype=np.float32)
    y_position = np.ones(total_localizations, dtype=np.float32)
    s_dev_x = np.ones(total_localizations, dtype=np.float32)
    s_dev_y = np.ones(total_localizations, dtype=np.float32)
    com_px = np.ones(total_localizations, dtype=np.float32)
    com_py = np.ones(total_localizations, dtype=np.float32)

    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        events_group = events[(events.group == g)]
        pick_photons = get_photons.get_pick_photons(events_group, photons,
                                                    drift, offset,
                                                    box_side_length=box_side_length,
                                                    int_time=int_time)
        all_events_photons = get_photons.photons_of_many_events(events_group,
                                                                pick_photons,
                                                                (box_side_length*1.5))

        print('__calibrate_peak__')
        peak_arrival_time = fitting.calibrate_peak_events(all_events_photons)
        print('peak arrival time is: ', peak_arrival_time, '_________________')

        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            one_loc = locs_group.iloc[i - counter]
            cylinder_photons = get_photons.crop_cylinder(one_loc, all_events_photons, offset,
            box_side_length, int_time)
            phot_loc = pd.DataFrame(data=cylinder_photons)

            bg_total = one_loc.bg * (box_side_length/2) * np.pi # calculates bg photons
            #my_bg_total = my_bg * (box_side_length/2) * np.pi # calculates new bg photons
            signal_photons = len(phot_loc) - bg_total
            #signal_photons_new = len(phot_loc) - my_bg_total

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            x, y , sd_x, sd_y = fitting.avg_of_roi(one_loc, one_loc.bg,
                                                   phot_loc, box_side_length,
                                                   return_sd=True)
            #my_x, my_y, my_sd_x, my_sd_y = fitting.avg_of_roi(one_loc, my_bg,
            #                                                  phot_loc, box_side_length,
            #                                                  return_sd=True)

            x_position[i] = x
            y_position[i] = y
            #my_x_position[i] = my_x
            #my_y_position[i] = my_y

            s_dev_x[i] = sd_x
            s_dev_y[i] = sd_y
            #my_s_dev_x[i] = my_sd_x
            #my_s_dev_y[i] = my_sd_y

            com_px[i] = fitting.localization_precision(signal_photons, sd_x, one_loc.bg)
            com_py[i] = fitting.localization_precision(signal_photons, sd_y, one_loc.bg)
            #my_com_px[i] = fitting.localization_precision(signal_photons_new, my_sd_x, my_bg)
            #my_com_py[i] = fitting.localization_precision(signal_photons_new, my_sd_y, my_bg)
            #my_bg_array[i] = my_bg


            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_loc,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)

    localizations['x'] = x_position.astype('float32')
    localizations['y'] = y_position.astype('float32')
    #localizations['x'] = my_x_position.astype('float32')
    #localizations['y'] = my_y_position.astype('float32')
    #localizations['my_bg'] = my_bg_array.astype('float32')
    localizations['sdx'] = s_dev_x.astype('float32')
    localizations['sdy'] = s_dev_y.astype('float32')
    #localizations['my_sdx'] = my_s_dev_x.astype('float32')
    #localizations['my_sdy'] = my_s_dev_y.astype('float32')
    localizations['com_px'] = com_px.astype('float32')
    localizations['com_py'] = com_py.astype('float32')
    #localizations['my_com_px'] = my_com_px.astype('float32')
    #localizations['my_com_py'] = my_com_py.astype('float32')
    localizations['lifetime'] = lifetime.astype('float32')
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file, '_lt_com')
    print(len(localizations), 'localizations tagged with lifetime and'
                              ' fitted with avg x,y position.')




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
    #my_s_dev_x = np.ones(total_localizations, dtype=np.float32)
    #my_s_dev_y = np.ones(total_localizations, dtype=np.float32)
    com_px = np.ones(total_localizations, dtype=np.float32)
    com_py = np.ones(total_localizations, dtype=np.float32)
    #my_com_px = np.ones(total_localizations, dtype=np.float32)
    #my_com_py = np.ones(total_localizations, dtype=np.float32)

    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        pick_photons = get_photons.get_pick_photons(locs_group, photons,
                                        drift, offset,
                                        box_side_length, integration_time)

        peak_arrival_time = fitting.calibrate_peak_locs(locs_group, pick_photons,
                                           offset, box_side_length,
                                           integration_time)
        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            one_loc = locs_group.iloc[i - counter]
            cylinder_photons = get_photons.crop_cylinder(one_loc, pick_photons, offset,
            box_side_length, integration_time)
            phot_loc = pd.DataFrame(data=cylinder_photons)

            bg_total = one_loc.bg * (box_side_length/2) * np.pi # calculates bg photons
            #my_bg_total = my_bg * (box_side_length/2) * np.pi # calculates new bg photons
            signal_photons = len(phot_loc) - bg_total
            #signal_photons_new = len(phot_loc) - my_bg_total

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            x, y , sd_x, sd_y = fitting.avg_of_roi(one_loc, one_loc.bg,
                                                   phot_loc, box_side_length,
                                                   return_sd=True)
            #my_x, my_y, my_sd_x, my_sd_y = fitting.avg_of_roi(one_loc, my_bg,
            #                                                  phot_loc, box_side_length,
            #                                                  return_sd=True)

            x_position[i] = x
            y_position[i] = y
            #my_x_position[i] = my_x
            #my_y_position[i] = my_y

            s_dev_x[i] = sd_x
            s_dev_y[i] = sd_y
            #my_s_dev_x[i] = my_sd_x
            #my_s_dev_y[i] = my_sd_y

            com_px[i] = fitting.localization_precision(signal_photons, sd_x, one_loc.bg)
            com_py[i] = fitting.localization_precision(signal_photons, sd_y, one_loc.bg)
            #my_com_px[i] = fitting.localization_precision(signal_photons_new, my_sd_x, my_bg)
            #my_com_py[i] = fitting.localization_precision(signal_photons_new, my_sd_y, my_bg)
            #my_bg_array[i] = my_bg


            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_loc,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)

    localizations['x'] = x_position.astype('float32')
    localizations['y'] = y_position.astype('float32')
    #localizations['x'] = my_x_position.astype('float32')
    #localizations['y'] = my_y_position.astype('float32')
    #localizations['my_bg'] = my_bg_array.astype('float32')
    localizations['sdx'] = s_dev_x.astype('float32')
    localizations['sdy'] = s_dev_y.astype('float32')
    #localizations['my_sdx'] = my_s_dev_x.astype('float32')
    #localizations['my_sdy'] = my_s_dev_y.astype('float32')
    localizations['com_px'] = com_px.astype('float32')
    localizations['com_py'] = com_py.astype('float32')
    #localizations['my_com_px'] = my_com_px.astype('float32')
    #localizations['my_com_py'] = my_com_py.astype('float32')
    localizations['lifetime'] = lifetime.astype('float32')
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file, '_lt_com_old')
    print(len(localizations), 'localizations tagged with lifetime and'
                              ' fitted with avg x,y position.')






def locs_lt_to_picasso_80(localizations_file, photons_file,
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
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = helper.process_input(drift_file, dataset='drift')
    lifetime = np.ones(len(localizations))
    lt_photons = np.ones(len(localizations), dtype=int)
    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        pick_photons = get_photons.get_pick_photons(
            locs_group, photons,
            drift, offset,
            box_side_length, integration_time)

        #peak_arrival_time = fitting.calibrate_peak(locs_group, pick_photons,
        #    offset, box_side_length,
        #    integration_time)

        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')
            phot_loc = get_photons.photons_of_one_localization(locs_group.iloc[i - counter],
                                                               pick_photons, offset,
                                                               box_side_length, integration_time)
            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))
            lifetime[i] = fitting.avg_lifetime_sergi_80(phot_loc, 80, 0)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')


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

        peak_arrival_time = fitting.calibrate_peak_locs(
            locs_group, pick_photons,
            offset, box_side_length,
            integration_time)

        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):

            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            phot_loc = get_photons.photons_of_one_localization(
                locs_group.iloc[i - counter],
                pick_photons, offset,
                box_side_length, integration_time)

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_loc,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_loc)

        counter += len(locs_group)


    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
