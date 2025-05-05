import numpy as np
from scipy.spatial import cKDTree


def local_brightness_map(events, radius=10, p=1, grid_size=1):
    """
    Compute a background (height) map from sparse localization data using inverse distance weighting (IDW)
    over a fixed neighborhood radius.

    For every grid pixel (with spacing defined by grid_size), this function finds all events
    within the given radius (in pixel units) and computes the weighted average of their brightness values, where the
    weight for each point is 1/(distance**p).

    Parameters
    ----------
    events : DataFrame or object with attributes 'x_array', 'y_array', and 'brightness_phot_ms'
        Sparse data points with x_array and y_array coordinates and corresponding background values.
    radius : float, optional
        The radius (in same units as x_array and y_array, e.g. pixels) within which to consider points.
        Default is 3.
    p : float, optional
        Power parameter for inverse distance weighting. Default is 1.
    grid_size : float, optional
        Spacing between grid points (default is 1, i.e. one pixel).

    Returns
    -------
    bg_map : 2D numpy.ndarray
        The computed background height map.
    grid_x : 1D numpy.ndarray
        The x_array coordinates corresponding to the grid.
    grid_y : 1D numpy.ndarray
        The y_array coordinates corresponding to the grid.
    """
    # Convert localization data to numpy arrays
    x = np.asarray(events.x, dtype=np.float64)
    y = np.asarray(events.y, dtype=np.float64)
    brightness_array = np.asarray(events.brightness_phot_ms, dtype=np.float64)

    # Determine grid extents (use floor for min and ceil for max)
    min_x, max_x = 0, 256#int(np.ceil(x_array.max())+2*radius)
    min_y, max_y = 0, 256#int(np.ceil(y_array.max())+2*radius)

    # Create grid coordinates (assuming grid points are at integer coordinates)
    grid_x = np.arange(min_x, max_x + 1, grid_size)
    grid_y = np.arange(min_y, max_y + 1, grid_size)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')

    # Build a KDTree for fast spatial queries on the localization points.
    points = np.column_stack((x, y))
    tree = cKDTree(points)

    # Initialize the background map
    brightness_map = np.zeros_like(X, dtype=np.float64)

    # Loop over every grid pixel and evaluate all points within the given radius.
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            grid_point = np.array([X[i, j], Y[i, j]])
            # Find indices of all localization points within the radius
            indices = tree.query_ball_point(grid_point, r=radius)
            if indices:
                # Compute Euclidean distances from the grid pixel to each selected point
                distances = np.sqrt((x[indices] - grid_point[0]) ** 2 + (y[indices] - grid_point[1]) ** 2)
                # If one or more localizations fall exactly on the grid point, use their background value(s) directly.
                if np.any(distances == 0):
                    brightness_map[i, j] = np.mean(brightness_array[indices][distances == 0])
                else:
                    # Compute weights as 1/(distance**p)
                    weights = 1 / (distances ** p)
                    weighted_brightness = np.sum(brightness_array[indices] * weights)
                    sum_weights = np.sum(weights)
                    brightness_map[i, j] = weighted_brightness / sum_weights
            else:
                # If no localization points are found within the radius, assign a NaN (or set to 0 as desired)
                brightness_map[i, j] = 0.01

    return brightness_map, grid_x, grid_y


# Example usage:
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    filename = 't/orig58_all_f_event_bright_norm.hdf5'

    events = pd.read_hdf(filename, key='locs')

    # Compute the background map using a radius of 3 pixels.
    bg_map, grid_x, grid_y = local_brightness_map(events, radius=5, p=0.5, grid_size=1)

    print(f'type of bg_map: {type(bg_map)}')
    print(f'shape of bg_map: {bg_map.shape}')

    print(f'type of grid_x: {type(grid_x)}')
    print(f'shape of grid_x: {grid_x.shape}')

    print(f'type of grid_y: {type(grid_y)}')
    print(f'shape of grid_y: {grid_y.shape}')


    # Plot the resulting height map
    plt.figure(figsize=(10, 8))
    plt.imshow(bg_map, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Background intensity')
    plt.title('Background Height Map (IDW over 3-pixel neighborhood)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()