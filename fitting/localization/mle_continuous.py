import numpy as np

def mle_continuous(
    x_array, y_array,
    x_start, y_start,
    diameter,
    bind_time_ms=200,
    isotropic=True,
    f_init=0.9,
    f_min=0.1,
    max_iter=100,
    tol=1e-4
):
    """
    Fit a 2D Gaussian + uniform background via EM on photon coords,
    using weights instead of cropping to ROI.

    Photon coords outside the circular ROI are given zero weight each iteration.

    Returns:
      mu_x, mu_y, sigma_x, sigma_y, f,
      tot_photons (in ROI), signal_photons, bg_photons, bg_rate
    """
    # Convert to numpy arrays
    x = np.asarray(x_array, float)
    y = np.asarray(y_array, float)
    if x.shape != y.shape:
        raise ValueError("x_array and y_array must have same shape")

    radius = diameter / 2.0
    fit_area = np.pi * radius**2
    N_total = x.size

    # Initialize parameters
    mu_x, mu_y = float(x_start), float(y_start)
    f = float(f_init)

    # Precompute constant for background pdf inside ROI
    p_bg_const = 1.0 / fit_area

    for _ in range(max_iter):
        # 1) compute mask for ROI (weights 1 inside, 0 outside)
        dx = x - mu_x
        dy = y - mu_y
        mask = (dx*dx + dy*dy) <= radius**2
        N_roi = mask.sum()
        if N_roi == 0:
            raise ValueError(f"No photons in ROI radius {radius}")

        # 2) expectation step: compute responsibilities
        # Gaussian pdf
        # ensure sigmas are positive
        sigma_x = max(1e-12, sigma_x if 'sigma_x' in locals() else np.std(x[mask]))
        sigma_y = max(1e-12, sigma_y if 'sigma_y' in locals() else np.std(y[mask]))
        if isotropic:
            avg_var = 0.5 * (sigma_x**2 + sigma_y**2)
            sigma_x = sigma_y = np.sqrt(max(avg_var, 1e-12))

        norm = 2 * np.pi * sigma_x * sigma_y
        G = np.exp(-0.5*((dx/sigma_x)**2 + (dy/sigma_y)**2)) / norm

        # Background pdf only inside ROI
        p_bg = p_bg_const * mask.astype(float)

        mix_sig = f * G * mask.astype(float)
        mix_bg = (1 - f) * p_bg
        denom = mix_sig + mix_bg
        # responsibilities: avoid division by zero
        w = np.zeros_like(denom)
        nonzero = denom > 0
        w[nonzero] = mix_sig[nonzero] / denom[nonzero]
        W = w.sum()

        # 3) maximization step
        mu_x_new = (w * x).sum() / W
        mu_y_new = (w * y).sum() / W

        if isotropic:
            var = (w * (dx**2 + dy**2)).sum() / (2 * W)
            sigma_x_new = sigma_y_new = np.sqrt(max(var, 1e-12))
        else:
            sigma_x_new = np.sqrt((w * dx**2).sum() / W)
            sigma_y_new = np.sqrt((w * dy**2).sum() / W)

        f_new = max(W / N_roi, f_min)

        # check convergence
        if (abs(mu_x_new - mu_x) < tol and
            abs(mu_y_new - mu_y) < tol and
            abs(sigma_x_new - sigma_x) < tol and
            abs(sigma_y_new - sigma_y) < tol and
            abs(f_new - f) < tol):
            mu_x, mu_y = mu_x_new, mu_y_new
            sigma_x, sigma_y, f = sigma_x_new, sigma_y_new, f_new
            break

        mu_x, mu_y = mu_x_new, mu_y_new
        sigma_x, sigma_y, f = sigma_x_new, sigma_y_new, f_new

    # compute final counts
    tot_photons = N_roi
    signal_photons = W
    bg_photons = tot_photons - signal_photons
    bg_rate = bg_photons * (200 / (bind_time_ms * fit_area))

    return {
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'f': f,
        'tot_photons': tot_photons,
        'signal_photons': signal_photons,
        'bg_photons': bg_photons,
        'bg_rate': bg_rate
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