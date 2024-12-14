import numpy as np
import pandas as pd
import helper
import fitting
import get_photons



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
    localizations = pd.read_hdf(localizations_file, key='locs')
    total_localizations = len(localizations)
    photons = pd.read_hdf(photons_file, key='photons')
    print(len(photons), ' photons and ', total_localizations,
          'localization read in')
    drift = helper.process_input(drift_file, dataset='drift')


    lifetime = np.ones(total_localizations)
    lt_photons = np.ones(total_localizations, dtype=int)
    x_position = np.ones(total_localizations)
    y_position = np.ones(total_localizations)
    s_dev_x = np.ones(total_localizations)
    s_dev_y = np.ones(total_localizations)

    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')

        pick_photons = get_photons.get_pick_photons(locs_group, photons,
                                        drift, offset,
                                        box_side_length, integration_time)

        peak_arrival_time = fitting.calibrate_peak(locs_group, pick_photons,
                                           offset, box_side_length,
                                           integration_time)

        # iterating over every localization in pick
        for i in range(counter, counter + len(locs_group)):
            if i == 0: print('fitting lifetime of ', len(locs_group),
                             ' localizations.')

            one_loc = locs_group.iloc[i - counter]
            phot_loc = pd.DataFrame(data=get_photons.crop_cylinder
                (one_loc, pick_photons, offset,
                box_side_length, integration_time))

            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))

            x, y , sd_x, sd_y = fitting.avg_of_roi(one_loc, phot_loc, box_side_length, return_sd=True)

            x_position[i] = x
            y_position[i] = y
            s_dev_x[i] = sd_x
            s_dev_y[i] = sd_y

            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_loc,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)
    localizations['x'] = x_position
    localizations['y'] = y_position
    localizations['sdx'] = s_dev_x
    localizations['sdy'] = s_dev_y
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    helper.dataframe_to_picasso(
        localizations, localizations_file, '_lt_sd')
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

        peak_arrival_time = fitting.calibrate_peak(
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
