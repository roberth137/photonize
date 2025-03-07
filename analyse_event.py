import event
import time


start_time = time.time()
folder = 'data/ml/single/'
event.event_analysis(localizations_file=f'{folder}a550_fp.hdf5',
                     photons_file=f'{folder}a550_index.hdf5',
                     drift_file=f'{folder}a550_drift.txt',
                     offset=10,
                     diameter=4.5,
                     int_time=200,
                     suffix='_just_bg',
                     max_dark_frames=1,
                     proximity=2,
                     filter_single=True,
                     norm_brightness=True,
                     dt_window=[0,2500])
end_time = time.time()
print(f"Execution time: {end_time-start_time:.3f} seconds")

