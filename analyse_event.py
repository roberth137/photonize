import event
import time


start_time = time.time()
event.event_analysis(localizations_file='ml/event_data/a550_fp.hdf5',
                     photons_file='ml/event_data/a550_index.hdf5',
                     drift_file='ml/event_data/a550_drift.txt',
                     offset=10,
                     diameter=4.5,
                     int_time=200,
                     suffix='_tailcut2500',
                     max_dark_frames=1,
                     proximity=2,
                     tailcut=2500)
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

