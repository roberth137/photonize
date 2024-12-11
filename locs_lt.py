import numpy as np
import pandas as pd
import core
import get_photons
import helper
import fitting



def locs_lt_avg_pos(localizations_file, photons_file,
                    drift_file, offset, box_side_length=5,
                    integration_time=200):
    '''
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
    x_position = np.ones(len(localizations))
    y_position = np.ones(len(localizations))
    counter = 0
    # iterating over every pick in file
    for g in set(localizations['group']):
        locs_group = localizations[(localizations.group == g)]
        print(len(locs_group), 'localizations in current group.')
        pick_photons = get_pick_photons(locs_group, photons,
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
            phot_loc = pd.DataFrame(data=core.crop_cylinder
            (one_loc, pick_photons, offset,
             box_side_length, integration_time))
            if i % 200 == 0: print('200 fitted. Number of photons',
                                   ' in last fit: ', len(phot_loc))
            x, y = fitting.avg_of_roi(one_loc, phot_loc, box_side_length)
            x_position[i] = x
            y_position[i] = y
            lifetime[i] = fitting.avg_lifetime_sergi_40(phot_loc,
                                                       peak_arrival_time)
            lt_photons[i] = len(phot_loc)
        counter += len(locs_group)
    localizations['x'] = x_position
    localizations['y'] = y_position
    localizations['lifetime'] = lifetime
    localizations['lt_photons'] = lt_photons
    core.dataframe_to_picasso(
        localizations, localizations_file, '_lt_avgPos_noBg')
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
        pick_photons = get_pick_photons(locs_group, photons,
                                        drift, offset,
                                        box_side_length, integration_time)
        # peak_arrival_time = calibrate_peak(locs_group, pick_photons,
        # offset, box_side_length,
        # integration_time)
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
    core.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')



def get_pick_photons(
        locs_group, photons, drift, offset,
        box_side_length, integration_time):
    '''
    Parameters
    ----------
    locs_group : localizations of this pick (group) as pd dataframe
    photons : photons as pd dataframe
    drift : drift as pd dataframe
    integration time: camera integration time
    box_side_length: size of the PSF in pixels

    Returns
    -------
    All driftcorrected photons in the area
    of the pick +- box_side_length/2
    '''
    # set dimensions of the region and crop photons
    # -0.53125 because: -> see undrift (pixel conversion)
    dr_x, dr_y = max(abs(drift.x)), max(abs(drift.y))
    min_x, max_x, min_y, max_y = core.min_max_box(locs_group, box_side_length)
    phot_cr = core.crop_photons(photons,
                                (min_x - 0.53125 - dr_x),
                                (max_x - 0.53125 + dr_x),
                                (min_y - 0.53125 - dr_y),
                                (max_y - 0.53125 + dr_y))
    print('number of cropped photons: ', len(phot_cr))
    # undrift photons
    phot_cr_und = core.undrift(phot_cr, drift, offset, integration_time)
    # crop photons again after drift
    phot_cr_und_cr = core.crop_photons(phot_cr_und,
                                       min_x, max_x, min_y, max_y)
    print('number of cropped-undrifted-cropped photons: ',
          len(phot_cr_und_cr))
    return phot_cr_und_cr


def loc_boundaries(localization, offset,
                   box_side_length, integration_time):
    x_min = localization.x - (box_side_length / 2)
    x_max = x_min + box_side_length
    y_min = localization.y - (box_side_length / 2)
    y_max = y_min + box_side_length
    ms_min = (localization.frame / offset) * integration_time
    ms_max = ms_min + integration_time
    return x_min, x_max, y_min, y_max, ms_min, ms_max


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
        pick_photons = get_photons.crop_undrift_crop(locs_group, photons,
                                                     drift, offset,
                                                     box_side_length, integration_time)
        peak_arrival_time = fitting.calibrate_peak(locs_group, pick_photons,
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
    core.dataframe_to_picasso(
        localizations, localizations_file)
    print(len(localizations), 'localizations tagged with lifetime')
