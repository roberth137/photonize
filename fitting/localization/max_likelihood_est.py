import numpy as np
import scipy.optimize as opt

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