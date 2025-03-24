import numpy as np
import utilities
from fitting.illumination.inverse_distance_bg import compute_bg_map_idw_radius

def normalize_brightness(events):
    """
    Normalizes brightness and bg of events using bg data. Uses inverse distance weighting

    Input:
        - events, pd DataFrame with column "bg"
        - method, which method to use, default is inverse distance weighting
    Output:
        - events with additional columns: 'bg_norm', 'bright_norm', 'lt_over_bright'
    """

    bg_map, grid_x, grid_y = compute_bg_map_idw_radius(events, radius=5, p=1, grid_size=1)

    print(f'mean bg values: {np.mean(events.bg)}')

    px_x = np.round(events['x']).astype(int)
    px_y = np.round(events['y']).astype(int)

    # In case some events are outside the computed map, clip indices to valid range.
    max_x, max_y = bg_map.shape
    px_x = np.clip(px_x, 0, max_x - 1)
    px_y = np.clip(px_y, 0, max_y - 1)

    # bg_map takes first y then x coordinate... careful!
    # Extract normalization values from the smoothed background map
    norm_values = bg_map[px_y, px_x]

    # Prevent division by zero by replacing any zeros with 1 (or handle as appropriate)
    norm_values_safe = np.where(norm_values == 0, 1, norm_values)

    # Normalize the background and brightness

    events['bg_200ms_px_norm'] = (events['bg_200ms_px']/norm_values_safe).astype(np.float32)
    events['brightness_norm'] = (events['brightness_phot_ms']/norm_values_safe).astype(np.float32)
    events['lt_over_bright'] = (events['lifetime_10ps']/events['brightness_norm']).astype(np.float32)
    if hasattr(events, 'bg_picasso'):
        events['bg_pic_norm'] = (events['bg_picasso']/norm_values_safe).astype(np.float32)


    return events

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    filename = 't/orig58_pf_event.hdf5'
    # Example: Create a DataFrame of localization points

    events = pd.read_hdf(filename, key='locs')

    events = normalize_brightness(events)

    utilities.dataframe_to_picasso(events, filename, '_main_norm_bright')

    # Compute the background map using a radius of 3 pixels.
    bg_map, grid_x, grid_y = compute_bg_map_idw_radius(events, radius=5, p=1, grid_size=1)

    plt.figure(figsize=(6, 5))
    plt.imshow(bg_map, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Background intensity')
    plt.title('Background Height Map (IDW over 3-pixel neighborhood)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()





