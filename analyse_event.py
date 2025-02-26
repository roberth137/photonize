import event
import time


start_time = time.time()
event.event_analysis(localizations_file='t/orig58_pf.hdf5',
                     photons_file='t/orig58_index.hdf5',
                     drift_file='t/orig58_drift.txt',
                     offset=10,
                     diameter=4.5,
                     int_time=200,
                     suffix='LQ_bs100',
                     max_dark_frames=1,
                     proximity=2)
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

