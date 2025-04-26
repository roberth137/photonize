import numpy as np


def mle_2d_gaussian_with_bg(coords, approx_mu, radius,
                                     isotropic=True, f_init=0.9, f_min=0.1,
                                     max_iter=100, tol=1e-6):
    """
    Fit a 2D Gaussian plus uniform background to raw photon coordinates
    within a circular ROI via the EM algorithm.

    Parameters
    ----------
    coords : array-like, shape (N,2)
        Photon (x,y) coordinates in continuous units (e.g., subpixels).
    approx_mu : tuple of float
        Initial guess for the Gaussian center (mu_x, mu_y).
    radius : float
        Radius of the circular ROI around approx_mu to include photons.
    isotropic : bool, optional
        If True (default), fit a single sigma for both axes;
        otherwise fit independent sigma_x and sigma_y.
    f_init : float in (0,1), optional
        Initial fraction of photons assigned to the Gaussian component (default 0.9).
    f_min : float in [0,1], optional
        Minimum allowed Gaussian fraction to prevent background overestimation
        (default 0.1).
    max_iter : int, optional
        Maximum number of EM iterations (default 100).
    tol : float, optional
        Convergence threshold for parameter changes (default 1e-6).

    Returns
    -------
    result : dict
        Fitted parameters:
            mu_x, mu_y       : float, Gaussian center coordinates
            sigma_x, sigma_y : float, fitted Gaussian widths
            f                : float, final fraction of signal photons
            n_photons        : int, number of photons in ROI
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N,2), got {coords.shape}")

    # Initial center for ROI
    mu_x, mu_y = approx_mu
    dx = coords[:, 0] - mu_x
    dy = coords[:, 1] - mu_y
    mask = dx*dx + dy*dy <= radius*radius
    coords = coords[mask]

    N = len(coords)
    if N == 0:
        raise ValueError(f"No photons inside circular ROI of radius {radius}")

    x = coords[:, 0]
    y = coords[:, 1]
    # Area of circle = pi * r^2
    S = np.pi * (radius**2)

    # Initialize widths and mixing fraction
    if isotropic:
        sigma = float(np.std(coords, axis=0).mean())
        sigma_x = sigma_y = sigma
    else:
        sigma_x = float(np.std(x))
        sigma_y = float(np.std(y))
    f = float(f_init)

    # EM iterations
    for _ in range(int(max_iter)):
        # Prevent divide-by-zero
        sigma_x = max(sigma_x, 1e-12)
        sigma_y = max(sigma_y, 1e-12)

        # Compute Gaussian PDF values at each photon
        norm = 2 * np.pi * sigma_x * sigma_y
        exp_x = np.exp(-0.5 * ((x - mu_x)**2) / (sigma_x**2))
        exp_y = np.exp(-0.5 * ((y - mu_y)**2) / (sigma_y**2))
        G = (exp_x * exp_y) / norm
        p_bg = 1.0 / S

        # E-step: responsibilities
        mix_sig = f * G
        mix_bg  = (1.0 - f) * p_bg
        denom   = mix_sig + mix_bg
        w       = mix_sig / denom

        # M-step: weighted updates
        W = w.sum()
        mu_x_new = (w * x).sum() / W
        mu_y_new = (w * y).sum() / W

        if isotropic:
            var = (w * ((x - mu_x_new)**2 + (y - mu_y_new)**2)).sum() / (2 * W)
            sigma_x_new = sigma_y_new = np.sqrt(max(var, 1e-12))
        else:
            sigma_x_new = np.sqrt((w * (x - mu_x_new)**2).sum() / W)
            sigma_y_new = np.sqrt((w * (y - mu_y_new)**2).sum() / W)

        f_new = max(W / N, f_min)

        # Check convergence
        if (abs(mu_x_new - mu_x) < tol and
            abs(mu_y_new - mu_y) < tol and
            abs(sigma_x_new - sigma_x) < tol and
            abs(sigma_y_new - sigma_y) < tol and
            abs(f_new - f) < tol):
            mu_x, mu_y = mu_x_new, mu_y_new
            sigma_x, sigma_y = sigma_x_new, sigma_y_new
            f = f_new
            break

        mu_x, mu_y = mu_x_new, mu_y_new
        sigma_x, sigma_y = sigma_x_new, sigma_y_new
        f = f_new

    return {
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'f': f,
        'n_photons': int(N)
    }