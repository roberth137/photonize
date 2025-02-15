import event
import time


start_time = time.time()
event.event_analysis(localizations_file='ml/backup/cy3_200ms_fp.hdf5',
                     photons_file='ml/event_data/cy3_59_index.hdf5',
                     drift_file='ml/event_data/cy3_200ms_drift.txt',
                     offset=10,
                     diameter=6,
                     int_time=200,
                     suffix='_diam6',
                     max_dark_frames=1,
                     proximity=2)
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

