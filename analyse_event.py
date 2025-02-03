import event
import time

#file = 't/orig58_all_f.hdf5'
#event.connect_locs_picasso_new(file, filter_single=False, proximity=2, max_dark_frames=2, suffix='2df')



start_time = time.time()
event.event_analysis('t/NUPS_pf_more.hdf5',
                     't/NUP_index.hdf5',
                     't/NUP_drift.txt',
                     10,
                     4.5,
                     200,
                     '_0mdf')
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

