import numpy as np
from typing import Optional
import warnings

def mle_fixed_sigma_bg(
    x_array, y_array,
    x_start: float,
    y_start: float,
    diameter: float,
    sigma: float,
    bg_rate: float,
    binding_time: float,
    max_iter: int = 100,
    tol: float = 1e-4,
    arrival_time: Optional[np.ndarray] = None
) -> dict:
    """
    EM-style fit of a 2D Gaussian center with fixed σ and adaptive background handling.

    This routine performs an Expectation-Maximization iteration to estimate the Gaussian
    center (mu_x, mu_y) given a fixed point-spread function width (sigma) and an
    initial background photon rate (bg_rate). If the expected number of background
    photons exceeds the total photons in the ROI, the background rate is halved and
    the EM continues, ensuring the fit remains stable.

    Parameters
    ----------
    x_array : array-like, shape (N,)
        x-coordinates of detected photons.
    y_array : array-like, shape (N,)
        y-coordinates of detected photons.
    x_start : float
        Initial guess for the Gaussian center x-coordinate.
    y_start : float
        Initial guess for the Gaussian center y-coordinate.
    diameter : float
        Diameter of the circular ROI (same spatial units as x,y).
    sigma : float
        Fixed standard deviation of the Gaussian PSF.
    bg_rate : float
        Initial background count rate (photons per pixel per unit time).
    binding_time : float
        Exposure or binding time in the same time units as bg_rate.
    max_iter : int, optional (default=100)
        Maximum number of EM iterations.
    tol : float, optional (default=1e-4)
        Convergence tolerance: if the center shift is less than tol, EM stops.

    Returns
    -------
    result : dict
        Dictionary containing:
          - 'mu_x': float, final fitted x-center
          - 'mu_y': float, final fitted y-center
          - 'weights': ndarray, P(signal | photon) for each photon in ROI
          - 'iters': int, number of iterations performed
          - 'bg_rate': float, possibly reduced background rate used in final fit

    Raises
    ------
    ValueError
        If no photons fall within the ROI, or the weight denominator becomes zero.

    Warnings
    --------
    RuntimeWarning
        - If expected background >= total photons, bg_rate is halved and a warning is issued.
        - If all weights become zero at some iteration, EM terminates early with a warning.
    """
    x = np.asarray(x_array, float)
    y = np.asarray(y_array, float)
    if arrival_time is not None: dt = np.asarray(arrival_time, float)
    radius = diameter / 2.0
    bg_rate = bg_rate * 2

    for it in range(1, max_iter + 1):
        # 1) Mask photons inside current ROI
        dx = x - x_start
        dy = y - y_start
        r = np.hypot(dx, dy)
        mask = r <= radius
        x_in = x[mask]
        y_in = y[mask]
        r_in = r[mask]
        dt_in = dt[mask]
        N = r_in.size

        if N == 0:
            raise ValueError(
                f"No photons within diameter={diameter:.3f} of ({x_start:.3f},{y_start:.3f})."
            )

        # 2) Compute expected background in ROI
        expected_bg = bg_rate * (binding_time / 200.0) * np.pi * radius**2

        if expected_bg >= N:
            warnings.warn(
                f"Expected background ({expected_bg:.1f}) >= total photons ({N}); "
                "halving bg_rate and retrying.",
                RuntimeWarning
            )
            bg_rate *= 0.5

            expected_bg = bg_rate * (binding_time / 200.0) * np.pi * radius**2

        # 3) Mixture priors
        N_sig = max(N - expected_bg, 0.0)
        P_sig = N_sig / N
        P_bg = 1.0 - P_sig

        # 4) PDFs for signal (Gaussian) and background (uniform)
        f_sig = np.exp(-r_in**2 / (2*sigma**2)) / (2*np.pi*sigma**2)
        area = np.pi * radius**2
        f_bg = 1.0 / area

        denom = P_sig * f_sig + P_bg * f_bg
        if np.all(denom == 0):
            raise ValueError("Denominator in weight computation is zero.")

        w = (P_sig * f_sig) / denom

        # 5) M-step: update center by weighted average
        w_sum = w.sum()
        if w_sum == 0:
            warnings.warn(
                f"All weights zero at iteration {it}; stopping EM.",
                RuntimeWarning
            )
            break

        mu_x_new = (w * x_in).sum() / w_sum
        mu_y_new = (w * y_in).sum() / w_sum

        # 6) Check convergence
        shift = np.hypot(mu_x_new - x_start, mu_y_new - y_start)
        x_start, y_start = mu_x_new, mu_y_new
        if shift < tol:
            break
    # How much is signal:
    print(f'Signal: {P_sig}, total photons:{N}')
    #if arrival_time: lifetime = (w * dt).sum() / w_sum
    result = {
    'mu_x': x_start,
    'mu_y': y_start,
    'weights': w,
    'iters': it,
    'bg_rate': bg_rate
    }
    if arrival_time is not None:
        lifetime = (w * dt_in).sum() / w_sum
        result['lifetime'] = lifetime

    return result
