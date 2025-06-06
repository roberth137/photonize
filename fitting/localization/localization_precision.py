import numpy as np

def localization_precision(sigma, photons, bg, pixel_nm):
    """
    Calculates localization precision, works with numpy arrays
    Adapted from Mortensen et al., Nat Methods 7, 377–381 (2010).

    LP = sigma_A^2 / photons * (16/9 + (8 * pi * sigma_A^2 * bg^2) / photons)

    sigma_A = (sigma^2 + 1/12 (pixel_115nm^2))

    Input: all numpy arrays
        - sigma (size of the PSF in either x or y)
        - photons (photons considered for fitting)
        - bg (background of the event, per pixel (here:115nm) per 200ms)
        - true pixel size (to be discussed)
    Output:
        - localization precision vector
    """
    pixel2 = (pixel_nm/115) ** 2
    s2 = sigma ** 2
    sa2 = s2 + pixel2 / 12
    variance = sa2 * (16 / 9 + (8 * np.pi * sa2 * bg) / photons) / photons
    return safe_sqrt(variance)


def safe_sqrt(variance):
    with np.errstate(invalid="ignore"):
        # This will produce NaN for negative values, instead of throwing a warning
        result = np.sqrt(variance)
    # Replace NaNs with 100
    result = np.nan_to_num(result, nan=100.0)
    return result
