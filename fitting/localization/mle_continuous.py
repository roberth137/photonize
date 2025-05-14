import numpy as np

def mle_continuous(x_array, y_array,
                   x_start, y_start,
                   diameter,
                   isotropic=True,
                   f_init=0.9,
                   f_min=0.1,
                   max_iter=100,
                   tol=1e-6):
    """
    Fit a 2D Gaussian + uniform background via EM on photon coords within a circle.

    Parameters
    ----------
    x_array, y_array : array-like, shape (N,)
        Photon coordinates.
    x_start, y_start : float
        Initial guess for Gaussian center.
    diameter : float
        ROI diameter around approx_mu.
    isotropic : bool
        If True, fit one sigma for both axes; else fit sigma_x, sigma_y.
    f_init : float
        Initial signal fraction (0<f_init<1).
    f_min : float
        Minimum allowed signal fraction.
    max_iter : int
        Max EM iterations.
    tol : float
        Convergence tolerance on parameter change.

    Returns
    -------
    dict with keys
      mu_x, mu_y       — fitted center
      sigma_x, sigma_y — fitted widths
      f                — final signal fraction
      tot_photons      — N in ROI
      signal_photons   — f⋅N
      bg_photons       — (1−f)⋅N
    """
    radius = diameter/2
    x_array = np.asarray(x_array, float)
    y_array = np.asarray(y_array, float)
    if x_array.shape != y_array.shape:
        raise ValueError("x_array and y_array must have same shape")


    # 3) EM loop
    for _ in range(max_iter):
        # 1) crop to ROI
        d2 = (x_array - x_start) ** 2 + (y_array - y_start) ** 2
        mask = d2 <= radius ** 2
        x_array = x_array[mask]
        y_array = y_array[mask]
        N = x_array.size
        if N == 0:
            raise ValueError(f"No photons in ROI radius {radius}")

        S = np.pi * radius ** 2  # area for uniform BG

        # 2) init parameters
        if isotropic:
            # variance across both dims
            vx = np.var(x_array)
            vy = np.var(y_array)
            sigma_x = sigma_y = np.sqrt(0.5 * (vx + vy))
        else:
            sigma_x = np.std(x_array)
            sigma_y = np.std(y_array)
        f = float(f_init)


        sigma_x = max(sigma_x, 1e-12)
        sigma_y = max(sigma_y, 1e-12)

        # Gaussian PDF at each photon (unnormalized)
        norm = 2*np.pi*sigma_x*sigma_y
        exp_x = np.exp(-0.5 * ((x_array - x_start) ** 2) / (sigma_x ** 2))
        exp_y = np.exp(-0.5 * ((y_array - y_start) ** 2) / (sigma_y ** 2))
        G = (exp_x * exp_y) / norm
        p_bg = 1.0 / S

        # responsibility for signal vs bg
        mix_sig = f * G
        mix_bg  = (1-f) * p_bg
        w = mix_sig / (mix_sig + mix_bg)
        W = w.sum()

        # M-step updates
        mu_x_new = (w * x_array).sum() / W
        mu_y_new = (w * y_array).sum() / W

        if isotropic:
            var = (w * ((x_array - mu_x_new) ** 2 + (y_array - mu_y_new) ** 2)).sum() / (2 * W)
            sigma_x_new = sigma_y_new = np.sqrt(max(var, 1e-12))
        else:
            sigma_x_new = np.sqrt((w * (x_array - mu_x_new) ** 2).sum() / W)
            sigma_y_new = np.sqrt((w * (y_array - mu_y_new) ** 2).sum() / W)

        f_new = max(W/N, f_min)

        # check convergence
        if (abs(mu_x_new-x_start)<tol and abs(mu_y_new-y_start)<tol and
            abs(sigma_x_new-sigma_x)<tol and abs(sigma_y_new-sigma_y)<tol and
            abs(f_new-f)<tol):
            x_start, y_start = mu_x_new, mu_y_new
            sigma_x, sigma_y, f = sigma_x_new, sigma_y_new, f_new
            break

        x_start, y_start = mu_x_new, mu_y_new
        sigma_x, sigma_y, f = sigma_x_new, sigma_y_new, f_new

    return {
        'mu_x': x_start,
        'mu_y': y_start,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'f': f,
        'tot_photons': N,
        'signal_photons':   N * f,
        'bg_photons':       N * (1 - f)
    }


def photon_signal_weights(r, diameter, sigma, background, total_photons):
    """
    Compute EM-style signal weights for photons in a circular ROI.

    Parameters
    ----------
    r : array-like, shape (M,)
        Radial distances of each of the M photons from the current center estimate.
    diameter : float
        Diameter of the circular fitting region (same units as r).
    sigma : float
        Fixed standard deviation of the 2D Gaussian PSF.
    background : float
        Fixed expected number of background photons in the region.
    total_photons : int
        Total number of photons observed in the region (signal + background).

    Returns
    -------
    w : ndarray, shape (M,)
        For each photon i, the weight w[i] = P(photon i ∈ signal | r[i], σ, B).
        These satisfy sum(w) ≈ total_photons – background.
    """
    # 1) Precompute mixture priors
    N_sig = total_photons - background
    p_sig = N_sig / total_photons
    p_bg = background / total_photons

    # 2) Densities:
    #   f_sig(r) = Gaussian PDF in 2D
    f_sig = np.exp(-r ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    #   f_bg  = uniform over the area
    area = np.pi * (diameter / 2) ** 2
    f_bg = 1.0 / area

    # 3) Posterior P(signal | r) for each photon
    w = (p_sig * f_sig) / (p_sig * f_sig + p_bg * f_bg)
    return w