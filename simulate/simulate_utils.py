import numpy as np
import simulate as s

def filter_points_by_radius(x, y, x_ref=s.x_ref, y_ref=s.y_ref, max_dist=None):
    """
    Calculate the distance from each (x[i], y[i]) to the given point.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    x_ref, y_ref
        The reference point (x0, y0).
    max_dist : max_dist to point

    Returns
    -------
    x: np.ndarray
    y: np.ndarray

    dist: 1D array of distances from each (x[i], y[i]) to 'point'.
    """

    dist = np.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2)

    if max_dist:
        mask = dist < max_dist
        return x[mask], y[mask], dist[mask]
    else:
        return x, y, dist

def freedman_diaconis_bins(data, data_range=None):
    data = np.asarray(data)
    n = len(data)
    if n <= 1:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2.0 * iqr / np.cbrt(n)
    if bin_width <= 0:
        return 1
    if data_range is None:
        data_min, data_max = data.min(), data.max()
    else:
        data_min, data_max = data_range
    bins = int(np.ceil((data_max - data_min) / bin_width))
    return max(bins, 1)