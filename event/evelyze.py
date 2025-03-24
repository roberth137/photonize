import numpy as np
import pandas as pd
from utilities import helper
from event import create_events
import fitting
import get_photons
from fitting import analyze_event

def event_analysis(localizations_file, photons_file, drift_file, offset,
                   diameter, int_time, suffix='', max_dark_frames=1,
                   proximity=2, filter_single=True, norm_brightness=False,
                   dt_window=None, more_ms=0, **kwargs):
    """

    reads in file of localizations, connects events and analyzes them

    """
    print('Starting event analysis: ...')
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')
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
    events = events_lt_avg_pos(events, photons, drift,
                               offset, diameter=diameter,
                               int_time=int_time, arrival_time=arrival_time,
                               dt_window=dt_window, more_ms=more_ms, **kwargs)
    if norm_brightness:
        print('Normalizing brightness...')
        events = fitting.normalize_brightness(events)
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
    total_events = len(events)
    photons = helper.process_input(photons_file, dataset='photons')
    print(f'starting events_lt_avg_pos... ')
    print(len(photons), ' photons and ', total_events,
          'events read in')
    drift = helper.process_input(drift_file, dataset='drift')

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
    start_dt = peak_arrival_time-0
    arrival_time['start'] = start_dt

    print('peak arrival time:   ', peak_arrival_time)
    print('start time:          ', start_dt)

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
        #print(f'only considering events with dt < 1700')

        # iterating over every event in pick
        for i in range(counter, counter + len(events_group)):
            my_event = events.iloc[i]

            cylinder_photons = get_photons.crop_event(my_event,
                                                      pick_photons,
                                                      diameter,
                                                      more_ms=more_ms)

            #######################################################################

            start_ms , end_ms, duration_ms = fitting.get_on_off_dur(cylinder_photons)

            #filter photons according to new bounds
            num_bg_photons = len(cylinder_photons[(cylinder_photons.ms > (my_event.start_ms_fr-more_ms))
                                              &(cylinder_photons.ms < (start_ms-20))
                                              |(cylinder_photons.ms < (my_event.end_ms_fr+more_ms))
                                              &(cylinder_photons.ms > (end_ms+20))])

            photons_new_bounds = cylinder_photons[(cylinder_photons.ms >= start_ms)
                                                  & (cylinder_photons.ms <= end_ms)]

            bg_measure_time = (end_ms+more_ms) - (start_ms-more_ms) - duration_ms - 40

            new_eve = pd.DataFrame([{'start_ms': start_ms, 'end_ms': end_ms, 'x': my_event.x, 'y': my_event.y}])
            this_eve = new_eve.iloc[0]
            photons_new_bounds = get_photons.crop_event(this_eve, pick_photons, diameter)

            total_photons = len(photons_new_bounds)
            phot_event = pd.DataFrame(data=photons_new_bounds)

            x_arr = phot_event['x'].to_numpy()
            y_arr = phot_event['y'].to_numpy()
            x_t, y_t, sd_x, sd_y = fitting.localize_com(x_arr,
                                                              y_arr,
                                                              return_sd=True)
            #calculate photon distances from new center for better lifetime determination:
            phot_x = np.copy(phot_event.x)
            phot_y = np.copy(phot_event.y)
            phot_x -= x_t
            phot_y -= y_t
            dist = np.sqrt((phot_x**2 + phot_y**2))
            phot_event['distance'] = dist
            arrival_times = phot_event['dt'].to_numpy()
            distance = phot_event['distance'].to_numpy()

            lifetime[i] = fitting.avg_lifetime_weighted(arrival_times,
                                                           distance,
                                                           start_dt,
                                                           diameter)

            result = analyze_event(cylinder_photons, start_dt, diameter)
            #########################################################
            x_position[i] = result.x_fit#x_t
            y_position[i] = result.y_fit#y_t
            sdx[i] = sd_x
            sdy[i] = sd_y
            total_photons_arr[i] = result.num_photons#total_photons
            start_ms_new[i] = result.start_ms#start_ms
            end_ms_new[i] = result.end_ms#end_ms
            duration_ms_arr[i] = result.duration_ms#duration_ms
            bg_200ms_pixel[i] = num_bg_photons*(200/bg_measure_time)/fit_area
            #bg_over_on[i] = len(cylinder_photons)/duration_ms
            delta_x[i] = my_event.x - result.x_fit#x_t
            delta_y[i] = my_event.y - result.y_fit#y_t
            # console printing
            if (i - counter) == 0:
                print('fitting lifetime of ', len(events_group),
                      ' events:')
            if i == 0:
                print('FIRST fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
            elif i % 200 == 0:
                print('200 fitted. Number of photons',
                      ' in phot_event: ', len(phot_event))
        counter += len(events_group)

    # calculate clipped bg
    bg_percentile = np.percentile(bg_200ms_pixel, 95)
    bg_200ms_pixel_capped = np.clip(bg_200ms_pixel, None, bg_percentile)

    squareroot_photons = np.sqrt(total_photons_arr)
    photons_arr = total_photons_arr - (bg_200ms_pixel*(duration_ms_arr/200)*fit_area)

    sx_arr = np.copy(events.sx)
    sy_arr = np.copy(events.sy)
    lpx_arr = np.copy(events.lpx)
    lpy_arr = np.copy(events.lpy)
    bg_picasso = np.copy(events.bg)


    events['x'] = x_position
    events['y'] = y_position
    events['photons'] = photons_arr.astype(np.int32)
    events['bg'] = bg_200ms_pixel_capped*duration_ms_arr/200
    events['lpx'] = fitting.localization_precision(sigma=sx_arr, photons=photons_arr, bg=bg_200ms_pixel_capped, pixel_nm=115)
    events['lpy'] = fitting.localization_precision(sigma=sy_arr, photons=photons_arr, bg=bg_200ms_pixel_capped, pixel_nm=115)
    events['lifetime_10ps'] = lifetime.astype(np.float32)
    events['bg_200ms_px'] = bg_200ms_pixel_capped.astype(np.float32)
    events['bg_picasso'] = bg_picasso.astype(np.float32)
    events['brightness_phot_ms'] = (photons_arr/duration_ms_arr).astype(np.float32)
    events['duration_ms'] = duration_ms_arr.astype(np.float32)
    events['lplt'] = (lifetime/photons_arr).astype(np.float32)
    events['start_ms'] = start_ms_new.astype(np.int32)
    events['end_ms'] = end_ms_new.astype(np.int32)
    events['delta_x'] = delta_x.astype(np.float32)
    events['delta_y'] = delta_y.astype(np.float32)
    events['bg_over_on'] = bg_200ms_pixel/duration_ms_arr.astype(np.float32)
    events['old_lpx'] = lpx_arr.astype(np.float32)
    events['old_lpy'] = lpy_arr.astype(np.float32)
    events.drop(columns=['start_ms_fr', 'end_ms_fr'], inplace=True)

    if isinstance(event_file, str):
        helper.dataframe_to_picasso(
            events, event_file, 'eve_lt_avgPos')
    print('__________________________FINISHED____________________________')
    print(f'\n{len(events)} events tagged with lifetime and'
                       ' fitted with avg x,y position.')
    return events
