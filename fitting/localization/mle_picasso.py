import numpy as np
from picasso import gaussmle

def fit_mle_picasso(spots, box_size, method="sigma", eps=1e-3, max_it=100):
    """
    Fit sub-pixel positions for a list of extracted spot patches using Picasso's gaussmle,
    returning coordinates relative to the patch origin (pixel (0,0) at its top-left corner).

    Parameters
    ----------
    spots : array-like, shape (N, box_size, box_size)
        Photon counts (or intensities) for each spot patch.
    box_size : int
        The side length of each square patch (e.g. 5).
    method : str, optional
        'sigma' (isotropic width) or 'sigmaxy' (independent widths). Default 'sigma'.
    eps : float, optional
        Convergence threshold in pixels. Default 1e-3.
    max_it : int, optional
        Maximum number of iterations per fit. Default 100.

    Returns
    -------
    results : list of dict
        For each spot, a dict with keys:
            x_rel, y_rel : float, sub-pixel coordinates relative to patch origin
            photons       : float, fitted total photons
            bg            : float, fitted background level
            sx, sy        : float, fitted sigma(s)
            lpx, lpy      : float, localization precision (CRLB)
            likelihood    : float, final log-likelihood
            iterations    : int, number of iterations until convergence
    """
    spots_arr = np.asarray(spots, dtype=np.float32)
    # Handle single patch
    single = False
    if spots_arr.ndim == 2:
        spots_arr = spots_arr[np.newaxis, ...]
        single = True
    elif spots_arr.ndim != 3:
        raise ValueError(f"spots must be 2D or 3D array, got shape {spots_arr.shape}")

    N, h, w = spots_arr.shape
    if h != box_size or w != box_size:
        raise ValueError(f"Each spot must be {box_size}×{box_size}, got {h}×{w}")

    # Run the MLE fitter (batch)
    thetas, CRLBs, liks, its = gaussmle.gaussmle(
        spots_arr, eps, max_it, method=method
    )

    results = []
    for i in range(N):
        theta = thetas[i]
        crlb = CRLBs[i]

        # Sub-pixel position relative to top-left corner
        x_rel = float(theta[0])
        y_rel = float(theta[1])

        photons = float(theta[2])
        bg = float(theta[3])
        if method == "sigmaxy":
            sx = float(theta[4])
            sy = float(theta[5])
        else:
            sx = sy = float(theta[4])

        lpx = float(np.sqrt(crlb[1]))
        lpy = float(np.sqrt(crlb[0]))

        results.append({
            'x_rel': x_rel,
            'y_rel': y_rel,
            'photons': photons,
            'bg': bg,
            'sx': sx,
            'sy': sy,
            'lpx': lpx,
            'lpy': lpy,
            'likelihood': float(liks[i]),
            'iterations': int(its[i]),
        })

    return results[0] if single else results


if __name__ == "__main__":
    # Example usage:
    # patches = [...]  # list or array of (5,5) patches
    # from fit_spots import fit_spot_patches
    # fits = fit_spot_patches(patches, box_size=5)
    # for f in fits:
    #     print(f)
    pass
