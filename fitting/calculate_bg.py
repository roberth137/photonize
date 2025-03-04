from math import floor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_laser_profile(localizations):
    """
    Returns parameters describing the laser profile using compute_bg_map and fit_gaussian_to_bg
    Parameters:
        localizations
    Returns:
        dict of gaussian parameters: return {
        "amplitude": amplitude,
        "x0": x0,
        "y0": y0,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "offset": offset,
        "fitted_map": fitted_gauss
    }
    """
    bg_map = compute_bg_map(localizations)
    fit_result = fit_gaussian_to_bg(bg_map)
    return fit_result

def normalize_brightness(events, laser_profile):
    """
    Takes in events and the laser profile and normalizes the 'bg' and 'brightness'
    of each event by dividing by the 2D Gaussian value at its (x, y) coordinate.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain columns: ['x', 'y', 'bg', 'brightness']
    laser_profile : dict
        Dictionary with keys:
           {
             "amplitude": float,
             "x0": float,
             "y0": float,
             "sigma_x": float,
             "sigma_y": float,
             "offset": float
           }

    Returns
    -------
    pd.DataFrame
        The same DataFrame with two new columns added:
        ['bg_norm', 'brightness_norm'].
    """

    # Extract the parameters from the laser_profile
    amplitude = laser_profile["amplitude"]
    x0 = laser_profile["x0"]
    y0 = laser_profile["y0"]
    sigma_x = laser_profile["sigma_x"]
    sigma_y = laser_profile["sigma_y"]
    offset = laser_profile["offset"]

    # Calculate the 2D Gaussian value for each event’s (x, y).
    # This is vectorized, so it works quickly on the entire columns.
    gauss_values = amplitude * np.exp(
        -(
            ((events['x'] - x0) ** 2) / (2 * sigma_x ** 2)
            + ((events['y'] - y0) ** 2) / (2 * sigma_y ** 2)
        )
    ) + offset

    # Add normalized columns to the events DataFrame:
    events['bg_norm'] = events['bg'] / gauss_values
    events['brightness_norm'] = events['brightness_phot_ms'] / gauss_values
    events['lt_over_bright'] = events['lifetime_10ps']/events['brightness_norm']

    return events

def compute_bg_map(localizations):
    """
    Create a 2D map of average background values from localization data.

    Parameters
    ----------
    localizations

    Returns
    -------
    bg_map : 2D numpy.ndarray
        A 2D array of average background values per pixel.
        Shape is determined by the max of the computed pixel indices.
    """
    x = np.asarray(localizations.x, dtype=np.float64)
    y = np.asarray(localizations.y, dtype=np.float64)
    bg = np.asarray(localizations.bg, dtype=np.float64)

    px_x = np.round(x).astype(int)
    px_y = np.round(y).astype(int)

    # Determine the shape of the output matrix
    max_x = px_x.max()
    max_y = px_y.max()
    # If indices start at 0, the size is max_index + 1
    bg_map = np.zeros((max_x + 1, max_y + 1), dtype=np.float64)
    count_map = np.zeros((max_x + 1, max_y + 1), dtype=np.int64)

    # Accumulate sums of background and counts using a fast NumPy approach
    np.add.at(bg_map, (px_x, px_y), bg)
    np.add.at(count_map, (px_x, px_y), 1)

    # Convert summed values to averages where count > 0
    mask = (count_map > 0)
    bg_map[mask] /= count_map[mask]

    return bg_map


def twoD_Gaussian(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    Elliptical 2D Gaussian function. We will flatten the 2D output
    so it can be used directly with scipy.curve_fit.

    coords : tuple of (x, y) meshgrid arrays, each flattened
    amplitude : peak value
    x0, y0 : center coordinates
    sigma_x, sigma_y : standard deviations along x and y
    offset : baseline intensity
    """
    x, y = coords
    g = amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2)
          + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    ) + offset
    return g.ravel()  # Flatten to 1D for curve_fit


def fit_gaussian_to_bg(bg_map):
    """
    Fit a 2D Gaussian to the background map (laser profile),
    considering only non-zero pixels from bg_map.

    Parameters
    ----------
    bg_map : 2D numpy.ndarray
        Background map where bg_map[i, j] is the average BG intensity
        at pixel (i, j). Zero means no localizations there (ignored).

    Returns
    -------
    result : dict
        Dictionary containing:
        - amplitude, x0, y0, sigma_x, sigma_y, offset (fitted params)
        - fitted_map (the 2D Gaussian model on the entire grid)
    """
    nx, ny = bg_map.shape
    # Create meshgrid with indexing="xy" => X.shape=(ny, nx), Y.shape=(ny, nx)
    x_vals = np.arange(nx)
    y_vals = np.arange(ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')  # (ny, nx)

    # Flatten
    x_data = X.ravel()  # shape = nx*ny
    y_data = Y.ravel()  # shape = nx*ny
    z_data = bg_map.T.ravel()  # transpose so shape matches (ny, nx)

    # ============================
    #  Filter out zero (or <=0) pixels
    # ============================
    # If you only want strictly positive, use z_data > 0
    # If you want non-zero (including negative?), use z_data != 0
    mask_nonzero = z_data > 0
    x_nz = x_data[mask_nonzero]
    y_nz = y_data[mask_nonzero]
    z_nz = z_data[mask_nonzero]

    # If there's nothing to fit, return None or raise an exception
    if len(z_nz) == 0:
        raise ValueError("All pixels are zero; nothing to fit.")

    # ==============
    #  Initial guesses
    # ==============
    amplitude_guess = np.nanmax(z_nz)  # peak
    offset_guess = np.nanmin(z_nz)  # baseline
    max_idx = np.nanargmax(z_nz)  # index of the maximum
    x0_guess = x_nz[max_idx]
    y0_guess = y_nz[max_idx]

    # A rough guess for sigmas ~ fraction of image dimensions
    sigma_x_guess = nx / 4.0
    sigma_y_guess = ny / 4.0

    initial_guess = (
        amplitude_guess,
        x0_guess,
        y0_guess,
        sigma_x_guess,
        sigma_y_guess,
        offset_guess,
    )

    # ==============
    #  Curve fit
    # ==============
    popt, pcov = curve_fit(
        twoD_Gaussian,
        (x_nz, y_nz),  # independent variables
        z_nz,  # data
        p0=initial_guess,
        maxfev=10000
    )

    amplitude, x0, y0, sigma_x, sigma_y, offset = popt

    # ==============
    #  Build the fitted 2D Gaussian on the full image grid
    # ==============
    # We want to produce a 2D array that matches bg_map shape
    fitted_gauss = twoD_Gaussian(
        (x_data, y_data),
        amplitude,
        x0,
        y0,
        sigma_x,
        sigma_y,
        offset
    ).reshape((ny, nx))  # Rebuild 2D, shape=(ny, nx)

    # By default, we used (x,y) = (X, Y) with shape (ny, nx),
    # but note that bg_map has shape (nx, ny). So if you prefer
    # the same orientation as bg_map, you can transpose here:
    fitted_gauss = fitted_gauss.T  # shape=(nx, ny)

    return {
        "amplitude": amplitude,
        "x0": x0,
        "y0": y0,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "offset": offset,
        "fitted_map": fitted_gauss
    }


# Example usage:
if __name__ == "__main__":
    filename = ('../local/4colors_2/picks.hdf5')
    localizations = pd.read_hdf(filename, key='locs')

    # Compute background map using 1.0 pixel size and floor indexing
    bg_map = compute_bg_map(localizations)

    # ------------------------------------------------------------------
    # 2. Fit a 2D Gaussian to the BG map
    # ------------------------------------------------------------------
    fit_result = fit_gaussian_to_bg(bg_map)
    fitted_map = fit_result["fitted_map"]

    print("Fitted parameters:")
    for k in ["amplitude", "x0", "y0", "sigma_x", "sigma_y", "offset"]:
        print(f"{k}: {fit_result[k]:.2f}")

    # ------------------------------------------------------------------
    # 3. Plot side by side: the raw BG map and the fitted Gaussian
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the raw BG map
    im1 = axes[0].imshow(
        bg_map.T, origin='lower', cmap='viridis', interpolation='nearest'
    )
    axes[0].set_title("Measured BG Map")
    axes[0].set_xlabel("X index")
    axes[0].set_ylabel("Y index")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Intensity')

    # Plot the fitted Gaussian
    im2 = axes[1].imshow(
        fitted_map.T, origin='lower', cmap='viridis', interpolation='nearest'
    )
    axes[1].set_title("Fitted Gaussian")
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Y index")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Intensity')

    plt.tight_layout()
    plt.show()

