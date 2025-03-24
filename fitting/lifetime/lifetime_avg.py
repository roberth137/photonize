import numpy as np

def avg_lifetime(loc_photons, peak):
    """
    Compute the average lifetime of photons, taking into account only those that arrive after a specified peak time.

    Parameters
    ----------
    loc_photons : pandas.DataFrame or a similar object
        The photon data, expected to have a column/attribute `dt` that contains
        the arrival times of the photons (in milliseconds, nanoseconds, or any
        other consistent time unit).
    peak : float or int
        The peak time. Photons arriving before this time are excluded from the
        lifetime calculation.

    Returns
    -------
    float
        The average lifetime of the photons arriving after `peak`.

    Notes
    -----
    - This function copies `loc_photons.dt` into a NumPy array to avoid
      unintentional modification of the input data.
    - Only the photon arrival times greater than `peak` are used in calculating
      the average. The lifetime is essentially `mean(arrival_time - peak)` for
      those times.
    - Ensure that `loc_photons.dt` is in a format that can be directly
      converted to a NumPy array (e.g., a pandas Series).
    """
    arr_times = np.copy(loc_photons.dt)
    arr_times_normalized = arr_times[arr_times > peak]
    lifetime = np.mean(arr_times_normalized - peak)
    return lifetime


def avg_lifetime_weighted(dt, distance, peak, diameter, base_weight=0.5, distance_weight=0.5):
    """
    Compute a weighted average lifetime using a quadratic distance-based weighting scheme.

    Parameters
    ----------
    dt : np.ndarray
        Time differences (e.g., photon arrival times) for each photon.
    distance : np.ndarray
        The distance of each photon from the localization center (e.g., the (x, y) event center).
    peak : float or int
        The peak time. Only photons arriving after this time are considered for the lifetime calculation.
    diameter : float
        The diameter of the localization region, used to determine the maximum distance for weighting.
    base : float, optional
        A baseline weight added to each photon’s weight regardless of distance. Default is 0.2.
    distance_weight : float, optional
        The scaling factor for the distance-based term. Default is 0.8.

    Returns
    -------
    np.float32
        The weighted average lifetime for photons arriving after `peak`.

    Notes
    -----
    - The weighting scheme is defined as:
      \[
      w_i = \text{base} + \text{distance_weight} \times
      \left(1 - \frac{ \text{distance}_i }{ \text{radius}^2 }\right)
      \]
      for each photon i, where `radius = diameter / 2`.
    - Only photons with `dt[i] > peak` are included in the sum.
    - When the total sum of weights is zero (e.g., no photons after `peak`),
      this function returns 0.0.
    - This function can be accelerated with Numba’s `njit` decorator (as you
      indicated in your comment).
    """
    radius = diameter / 2
    n = len(dt)

    weighted_sum = 0.0
    weight_total = 0.0

    for i in range(n):
        if dt[i] > peak:  # Only consider photons arriving after the peak
            ap_dt = dt[i] - peak
            # Quadratic weighting factor with a baseline offset
            ap_weight = base_weight + distance_weight * (1 - (distance[i]**2 / radius**2))
            weighted_sum += ap_dt * ap_weight
            weight_total += ap_weight

    lifetime = weighted_sum / weight_total if weight_total > 0 else 0.0
    return np.float32(lifetime)
