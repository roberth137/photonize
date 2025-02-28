import numpy as np
import numba
import scipy.optimize as opt



def avg_of_roi_cons_bg(localization, bg_pixel, phot_locs, box_side_length, return_sd=False):
    """
    Parameters
    ----------
    phot_locs : photons of one localization as pd dataframe
    Returns
    -------
    - x position
    - y position
    - sd_x
    - sd_y

    Position is calculated by adding up all photons in the circular
    surrounding of the localization.
    Background gets subtracted
    """
    if len(phot_locs)==0: print('avg_of_roi(), no photons: \n', localization)

    x_photons = phot_locs['x'].to_numpy()
    y_photons = phot_locs['y'].to_numpy()

    fit_area = np.pi * ((box_side_length / 2) ** 2)
    total_photons = len(phot_locs)
    bg_photons = bg_pixel * fit_area
    number_phot = (total_photons - bg_photons)

    pos_x = (np.sum(x_photons) - bg_photons * localization.x) / number_phot
    pos_y = (np.sum(y_photons) - bg_photons * localization.y) / number_phot

    if return_sd:

        sd_x = calculate_sd_cons_bg(x_photons, pos_x, number_phot, bg_photons, box_side_length)
        sd_y = calculate_sd_cons_bg(y_photons, pos_y, number_phot, bg_photons, box_side_length)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y

def avg_of_roi(localization, bg_pixel, phot_locs, box_side_length, return_sd=False):
    """
    Parameters
    ----------
    phot_locs : photons of one localization as pd dataframe
    Returns
    -------
    - x position
    - y position
    - sd_x
    - sd_y

    Position is calculated by adding up all photons in the circular
    surrounding of the localization.
    Background gets subtracted
    """
    if len(phot_locs)==0: print('avg_of_roi(), no photons: \n', localization)

    x_photons = phot_locs['x'].to_numpy()
    y_photons = phot_locs['y'].to_numpy()

    fit_area = np.pi * ((box_side_length / 2) ** 2)
    total_photons = len(phot_locs)
    bg_photons = bg_pixel * fit_area
    number_phot = (total_photons - bg_photons)

    pos_x = (np.sum(x_photons) - bg_photons * localization.x) / number_phot
    pos_y = (np.sum(y_photons) - bg_photons * localization.y) / number_phot

    if return_sd:

        sd_x = calculate_sd(x_photons, pos_x, total_photons)
        sd_y = calculate_sd(y_photons, pos_y, total_photons)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y



def calculate_sd_cons_bg(positions, pos_fit, number_phot, bg_total, diameter):
    """
    Calculates 1d std for a center of mass fit:
    positions: array with photons positions
    pos_fit: fitted position
    number_photons: total_photons - bg
    bg_total: total number of background photons
    diameter: diameter of roi
    """
    bg_var = ((diameter/4) ** 2) / 2 # s.d. in 1d of a random distribution on disk
    var = (np.sum((positions - pos_fit) ** 2) - (bg_var * bg_total)) / number_phot
    return 10 if var <= 0 else np.sqrt(var)

def calculate_sd_old(positions, pos_fit, total_phot):
    """
    Calculates 1d std for a center of mass fit, not considering bg
    positions: array with photons positions
    pos_fit: fitted position
    number_photons: total_photons - bg
    bg_total: total number of background photons
    diameter: diameter of roi
    """
    var = np.sum((positions - pos_fit) ** 2)/total_phot
    return 10 if var <= 0 else np.sqrt(var)

def sigma_sqrt_n(photons, sigma):
    """
    Calculates sigma/sqrt(N)
    """
    return (sigma / np.sqrt(photons))

def localization_precision(photons, sigma, bg):
    """
    Calculates the theoretical localization precision according to
    Mortensen et al., Nat Meth, 2010 for a 2D unweighted Gaussian fit.
    Copied from picasso
    """
    sigma2 = sigma**2
    sigma_a2 = sigma2 + 1 / 12
    v = sigma_a2 * (16 / 9 + (8 * np.pi * sigma_a2 * bg) / photons) / photons
    with np.errstate(invalid="ignore"):
        return np.sqrt(v)


def event_position_cons_bg(event, phot_event, diameter, return_sd=True):

    x_photons = phot_event['x'].to_numpy()
    y_photons = phot_event['y'].to_numpy()

    fit_area = np.pi * ((diameter / 2) ** 2)
    total_photons = len(phot_event)
    number_phot = (total_photons - fit_area * event.bg) #fit area in pixel^2 #bg per pixel
    bg = event.bg * fit_area
    #bg = 1 * fit_area

    pos_x = (np.sum(x_photons) - bg * event.x) / number_phot
    pos_y = (np.sum(y_photons) - bg * event.y) / number_phot

    if return_sd:
        sd_x = calculate_sd_cons_bg(x_photons, pos_x, number_phot, bg, diameter)
        sd_y = calculate_sd_cons_bg(y_photons, pos_y, number_phot, bg, diameter)

        return pos_x, pos_y, sd_x, sd_y

    else:
        return pos_x, pos_y

#numba.njit
def calculate_sd(photon_positions, mean_position, total_photons):
    return np.sqrt(np.sum((photon_positions - mean_position) ** 2) / total_photons)

#@numba.njit
def event_position(x_photons, y_photons, return_sd=True):
    #fit_area = np.pi * ((diameter / 2) ** 2)
    total_photons = len(x_photons)

    pos_x = np.sum(x_photons) / total_photons
    pos_y = np.sum(y_photons) / total_photons

    if return_sd:
        sd_x = calculate_sd(x_photons, pos_x, total_photons)
        sd_y = calculate_sd(y_photons, pos_y, total_photons)
        return pos_x, pos_y, sd_x, sd_y
    else:
        sd_x, sd_y = 0.0, 0.0
        return pos_x, pos_y, sd_x, sd_y


def gaussian_2d(params, x, y):
    """2D Gaussian function for MLE fitting."""
    x0, y0, sigma_x, sigma_y, amplitude, bg = params
    return amplitude * np.exp(
        -((x - x0) ** 2 / (2 * sigma_x ** 2)
          + (y - y0) ** 2 / (2 * sigma_y ** 2))
    ) + bg


def negative_log_likelihood(params, x_photons, y_photons):
    """Negative log-likelihood function for Gaussian MLE fitting."""
    # Unpack parameters
    x0, y0, sigma_x, sigma_y, amplitude, bg = params

    # Compute model
    model = gaussian_2d(params, x_photons, y_photons)

    # Compute negative log-likelihood (add small constant to avoid log(0))
    return -np.sum(np.log(model + 1e-9))


def event_position_mle(x_photons, y_photons, return_sd=True):
    """
    Perform MLE fitting to estimate event position (x, y) and standard deviations (sigma_x, sigma_y).

    Parameters:
    - x_photons: Array of x-coordinates of detected photons
    - y_photons: Array of y-coordinates of detected photons
    - return_sd: If True, returns standard deviations (sigma_x, sigma_y)

    Returns:
    - pos_x, pos_y: Estimated position of the event
    - sd_x, sd_y (optional): Estimated standard deviations in x and y
    """
    total_photons = len(x_photons)
    if total_photons == 0:
        return None, None, None, None  # No photons detected

    # Initial estimates: center of mass
    x0 = np.mean(x_photons)
    y0 = np.mean(y_photons)
    sigma_x0 = np.std(x_photons) if np.std(x_photons) > 0 else 1.0
    sigma_y0 = np.std(y_photons) if np.std(y_photons) > 0 else 1.0

    # A reasonable initial amplitude estimate:
    # total photons spread across a 2D Gaussian with initial sigmas
    amplitude0 = total_photons / (2.0 * np.pi * sigma_x0 * sigma_y0)
    bg0 = 0.0  # or use something like np.percentile(x_photons, 1) if you have a better BG guess

    # Parameter bounds to ensure realistic fitting
    # Each element is (lower_bound, upper_bound) for one parameter:
    param_bounds = [
        (x0 - 10, x0 + 10),  # x0
        (y0 - 10, y0 + 10),  # y0
        (0.01, 10.0),  # sigma_x
        (0.01, 10.0),  # sigma_y
        (1.0, total_photons * 2.0),  # amplitude
        (0.0, total_photons)  # background
    ]

    # Initial guesses
    p0 = [x0, y0, sigma_x0, sigma_y0, amplitude0, bg0]

    # Optimize using Maximum Likelihood Estimation (L-BFGS-B handles bounds)
    result = opt.minimize(
        negative_log_likelihood,
        x0=p0,
        args=(x_photons, y_photons),
        bounds=param_bounds,
        method='L-BFGS-B'
    )

    # If the optimizer fails, return initial estimates so it won't crash
    if not result.success:
        return x0, y0, sigma_x0, sigma_y0

    # Extract fitted parameters
    x_fit, y_fit, sigma_x_fit, sigma_y_fit, _, _ = result.x

    if return_sd:
        return x_fit, y_fit, sigma_x_fit, sigma_y_fit
    else:
        return x_fit, y_fit, 0.0, 0.0