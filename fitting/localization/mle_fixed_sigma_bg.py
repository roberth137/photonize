import numpy as np

def mle_fixed_sigma_bg(
    x_array, y_array,
    x_start, y_start,
    diameter,
    sigma,
    background,
    max_iter=100,
    tol=1e-6
):
    """
    EM‐style iteration to fit only the 2D Gaussian center with fixed σ and background.

    Parameters
    ----------
    x_array, y_array : array-like, shape (N,)
        Photon x,y coordinates.
    x_start, y_start : float
        Initial guess for Gaussian center.
    diameter : float
        ROI diameter around the center (same units as x,y).
    sigma : float
        Fixed Gaussian standard deviation.
    background : float
        Fixed expected number of background photons in the ROI.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on center shift.

    Returns
    -------
    dict with keys
      mu_x, mu_y   — fitted center coordinates
      weights      — final array of P(signal | r_i) for each photon in ROI
      iters        — number of iterations performed
    """
    x = np.asarray(x_array, float)
    y = np.asarray(y_array, float)
    radius = diameter / 2.0

    for it in range(1, max_iter + 1):
        # 1) Mask to ROI around current center
        dx = x - x_start
        dy = y - y_start
        r = np.hypot(dx, dy)
        mask = r <= radius
        x_in, y_in, r_in = x[mask], y[mask], r[mask]
        N = r_in.size
        if N == 0:
            raise ValueError(f"No photons within diameter {diameter} of ({x_start},{y_start})")

        # 2) Compute weights P(signal | r_i)
        # Mixture priors
        N_sig = N - background
        P_sig = max(N_sig, 0) / N
        P_bg  = 1 - P_sig

        # Signal PDF (2D Gaussian)
        f_sig = np.exp(-r_in**2 / (2*sigma**2)) / (2*np.pi*sigma**2)
        # Background PDF (uniform)
        area = np.pi * radius**2
        f_bg  = 1.0 / area

        w = (P_sig * f_sig) / (P_sig * f_sig + P_bg * f_bg)

        # 3) Update center
        mu_x_new = (w * x_in).sum() / w.sum()
        mu_y_new = (w * y_in).sum() / w.sum()

        # 4) Check convergence
        shift = np.hypot(mu_x_new - x_start, mu_y_new - y_start)
        x_start, y_start = mu_x_new, mu_y_new
        if shift < tol:
            break

    return {
        'mu_x': x_start,
        'mu_y': y_start,
        'weights': w,
        'iters': it
    }