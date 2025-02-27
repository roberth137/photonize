import event
import time


start_time = time.time()
event.event_analysis(localizations_file='t/third/picks.hdf5`',
                     photons_file='t/third/index.hdf5',
                     drift_file='t/third/drift.txt',
                     offset=10,
                     diameter=4.5,
                     int_time=200,
                     suffix='test_peak',
                     max_dark_frames=1,
                     proximity=2,
                     filter_single=True,
                     norm_brightness=True)
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

