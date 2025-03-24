import numpy as np
from scipy.spatial import cKDTree


def compute_bg_map_idw_radius(localizations, radius=10, p=1, grid_size=1):
    """
    Compute a background (height) map from sparse localization data using inverse distance weighting (IDW)
    over a fixed neighborhood radius.

    For every grid pixel (with spacing defined by grid_size), this function finds all localization points
    within the given radius (in pixel units) and computes the weighted average of their bg values, where the
    weight for each point is 1/(distance**p).

    Parameters
    ----------
    localizations : DataFrame or object with attributes 'x', 'y', and 'bg'
        Sparse data points with x and y coordinates and corresponding background values.
    radius : float, optional
        The radius (in same units as x and y, e.g. pixels) within which to consider points.
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
        The x coordinates corresponding to the grid.
    grid_y : 1D numpy.ndarray
        The y coordinates corresponding to the grid.
    """
    # Convert localization data to numpy arrays
    x = np.asarray(localizations.x, dtype=np.float64)
    y = np.asarray(localizations.y, dtype=np.float64)
    bg = get_bg(localizations)#np.asarray(localizations.bg, dtype=np.float64)

    # Determine grid extents (use floor for min and ceil for max)
    min_x, max_x = 0, int(np.ceil(x.max())+2*radius)
    min_y, max_y = 0, int(np.ceil(y.max())+2*radius)

    # Create grid coordinates (assuming grid points are at integer coordinates)
    grid_x = np.arange(min_x, max_x + 1, grid_size)
    grid_y = np.arange(min_y, max_y + 1, grid_size)
    X, Y = np.meshgrid(grid_x, grid_y, indexing='xy')

    # Build a KDTree for fast spatial queries on the localization points.
    points = np.column_stack((x, y))
    tree = cKDTree(points)

    # Initialize the background map
    bg_map = np.zeros_like(X, dtype=np.float64)

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
                    bg_map[i, j] = np.mean(bg[indices][distances == 0])
                else:
                    # Compute weights as 1/(distance**p)
                    weights = 1 / (distances ** p)
                    weighted_bg = np.sum(bg[indices] * weights)
                    sum_weights = np.sum(weights)
                    bg_map[i, j] = weighted_bg / sum_weights
            else:
                # If no localization points are found within the radius, assign a NaN (or set to 0 as desired)
                bg_map[i, j] = np.nan
            bg_map_x_y = bg_map#.T

    return bg_map_x_y, grid_x, grid_y

def get_bg(localizations):
    # Option 1: Check attribute existence:
    if hasattr(localizations, 'brightness_phot_ms') and localizations.brightness_phot_ms is not None:
        bg_array = np.asarray(localizations.brightness_phot_ms, dtype=np.float64)
    elif hasattr(localizations, 'bg_picasso') and localizations.bg_picasso is not None:
        bg_array = np.asarray(localizations.bg_picasso, dtype=np.float64)
        print(f'using bg_picasso column for normalization')
    else:
        bg_array = np.asarray(localizations.bg, dtype=np.float64)
        print(f'using bg column for normalization')
    return bg_array


# Example usage:
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    filename = 't/orig58_pf.hdf5'
    # Example: Create a DataFrame of localization points

    localizations = pd.read_hdf(filename, key='locs')

    # Compute the background map using a radius of 3 pixels.
    bg_map, grid_x, grid_y = compute_bg_map_idw_radius(localizations, radius=5, p=1, grid_size=1)

    print(f'type of bg_map: {type(bg_map)}')
    print(f'shape of bg_map: {bg_map.shape}')

    print(f'type of grid_x: {type(grid_x)}')
    print(f'shape of grid_x: {grid_x.shape}')

    print(f'type of grid_y: {type(grid_y)}')
    print(f'shape of grid_y: {grid_y.shape}')


    # Plot the resulting height map
    plt.figure(figsize=(6, 5))
    plt.imshow(bg_map, origin='lower', extent=(grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]),
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Background intensity')
    plt.title('Background Height Map (IDW over 3-pixel neighborhood)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()