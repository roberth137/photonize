import event
import time

#file = 't/orig58_all_f.hdf5'
#event.connect_locs_picasso_new(file, filter_single=False, proximity=2, max_dark_frames=2, suffix='2df')



start_time = time.time()
event.event_analysis(localizations_file='t/orig58_pf.hdf5',
                     photons_file='t/orig58_index.hdf5',
                     drift_file='t/orig58_drift.txt',
                     offset=10,
                     diameter=4.5,
                     int_time=200,
                     suffix='_test_params',
                     max_dark_frames=2,
                     proximity=1)
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

