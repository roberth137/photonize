import event
import time

start_time = time.time()
event.event_analysis('t/orig58_pf.hdf5',
                     't/orig58_index.hdf5',
                     't/orig58_drift.txt',
                     10,
                     4.5,
                     200,
                     'new_linking')
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

