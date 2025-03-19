import numpy as np
from fitting.illumination import compute_bg_map_idw_radius
import utilities

def normalize_brightness(events, method='idw'):
    """
    Normalizes brightness and bg of events using bg data.

    Input:
        - events, pd DataFrame with column "bg"
        - method, which method to use, default is inverse distance weighting
    Output:
        - events with additional columns: 'bg_norm', 'bright_norm', 'lt_over_bright'
    """
    #select method logic
    #if method=='idw':

    bg_map, grid_x, grid_y = compute_bg_map_idw_radius(events, 5, 1, 1)

    print(f'mean bg values: {np.mean(events.bg)}')

    px_x = np.round(events['x']).astype(int)
    px_y = np.round(events['y']).astype(int)

    # In case some events are outside the computed map, clip indices to valid range.
    max_x, max_y = bg_map.shape
    px_x = np.clip(px_x, 0, max_x - 1)
    px_y = np.clip(px_y, 0, max_y - 1)

    # Extract normalization values from the smoothed background map
    norm_values = bg_map[px_y, px_x]

    # Prevent division by zero by replacing any zeros with 1 (or handle as appropriate)
    norm_values_safe = np.where(norm_values == 0, 1, norm_values)

    print(f'max values bg_map: {np.nanmax(bg_map)}')

    num_non_nans = np.count_nonzero(~np.isnan(bg_map))

    print("Number of non-NaN values:", num_non_nans)

    mask = ~np.isnan(bg_map)

    # get indices
    indices = np.argwhere(mask)
    print(indices.shape)
    print(f'min max indices first {min(indices[:,0])},{max(indices[:,0])}')
    print(f'min max indices second {min(indices[:,1])},{max(indices[:,1])}')


    print("Indices of non-NaN values:")
    i=0
    for idx in indices:
        if i > 100:
            break
        i += 1
        print(tuple(idx))



    print(f'px_x\n{px_x[:10]}')
    print(f'px_y\n{px_y[:10]}')
    print(f'printing first bg_maps')
    for i in range(20):
        print(bg_map[px_y[i], px_x[i]])
        print(px_y[i], px_x[i])

    print(f'bg map: {bg_map}')
    print(f'shape bg_map: {bg_map.shape}')
    print(f'type of bg_map: {type(bg_map)}')
    print(f'shape of bg_map: {bg_map.shape}')
    print(f'type of grid_x: {type(grid_x)}')
    print(f'shape of grid_x: {grid_x.shape}')
    print(f'type of grid_y: {type(grid_y)}')
    print(f'shape of grid_y: {grid_y.shape}')


    # Normalize the background and brightness

    #events['bg_norm'] = (events['bg']/norm_values_safe).astype(np.float32) #/ norm_values_safe
    #the time scaled bg values should probably not be normalized

    events['bg_200ms_px_norm'] = (events['bg_200ms_px']/norm_values_safe).astype(np.float32) #/ norm_values_safe
    events['brightness_norm'] = (events['brightness_phot_ms']/norm_values_safe).astype(np.float32)
    events['lt_over_bright'] = (events['lifetime_10ps']/events['brightness_norm']).astype(np.float32)
    if hasattr(events, 'bg_picasso'):
        events['bg_pic_norm'] = (events['bg_picasso']/norm_values_safe).astype(np.float32)


    return events

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    filename = 't/orig58_pf_event_80more.hdf5'
    # Example: Create a DataFrame of localization points

    events = pd.read_hdf(filename, key='locs')

    # Compute the background map using a radius of 3 pixels.
    bg_map, grid_x, grid_y = compute_bg_map_idw_radius(events, radius=5, p=1, grid_size=1)

    events = normalize_brightness(events)

    utilities.dataframe_to_picasso(events, filename, '_main_norm_bright')

    plt.figure(figsize=(6, 5))
    plt.imshow(bg_map, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Background intensity')
    plt.title('Background Height Map (IDW over 3-pixel neighborhood)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()





