from utilities import helper
from event import create_events, analyze_events
import fitting

def evelyze(localizations_file, photons_file, drift_file, offset,
            diameter, int_time, suffix='', max_dark_frames=1,
            proximity=2, filter_single=True, norm_brightness=False,
            dt_window=None, more_ms=0, **kwargs):
    """

    reads in file of localizations, connects events and analyzes them

    """
    print('Starting event analysis: ...')
    # 1) read in files
    localizations = helper.process_input(localizations_file,
                                         dataset='locs')
    photons = helper.process_input(photons_file, dataset='photons')
    drift = helper.process_input(drift_file, dataset='drift')
    # 2) create preliminary events by linking localizations
    events = create_events.locs_to_events(localizations,
                                          offset=offset,
                                          int_time=int_time,
                                          max_dark_frames=max_dark_frames,
                                          proximity=proximity,
                                          filter_single=filter_single)
    # 3) analyze events in main loop (localization+lifetime+brightness)
    arrival_time = {}
    events = analyze_events.events_lt_pos(events, photons, drift,
                               offset, diameter=diameter,
                               int_time=int_time, arrival_time=arrival_time,
                               dt_window=dt_window, more_ms=more_ms, **kwargs)
    # 4) normalize brightness if applicable
    if norm_brightness:
        print('Normalizing brightness...')
        events = fitting.normalize_brightness(events)
    # 5) save events
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
