import numpy as np
import matplotlib.pyplot as plt
import simulate as s

def simulate_fluorophore(binding_time=s.binding_time_ms,
                         brightness=s.brightness,
                         sigma_psf=s.sigma_psf,
                         camera_error=s.camera_error,  # 20 microns error / 600 x magnification / 115nm pixel size
                         subpixel=s.subpixel):
    """
    Simulate one fluorophore at the origin (0,0),
    with photon positions sampled from a 2D Gaussian (sigma_psf).
    Then apply camera error and quantize to 'min_binning' steps.

    Parameters
    ----------
    num_photons : int
        Number of photons to emit.
    sigma_psf : float
        Standard deviation of the PSF (in pixel units).
    camera_error : float
        Additional camera noise (std in pixels).
    subpixel : float
        Minimum subpixel units of camera

    Returns
    -------
    x_fluo : np.ndarray
        Final x-coordinates of measured photon positions (after error + binning).
    y_fluo : np.ndarray
        Final y-coordinates of measured photon positions (after error + binning).
    """
    num_photons = int(binding_time * brightness)
    # 1) Ground truth photon positions
    x_photon = np.random.normal(loc=0.0, scale=sigma_psf, size=num_photons)
    y_photon = np.random.normal(loc=0.0, scale=sigma_psf, size=num_photons)

    # 2) Add camera error
    x_with_error = x_photon + np.random.normal(loc=0.0, scale=camera_error, size=num_photons)
    y_with_error = y_photon + np.random.normal(loc=0.0, scale=camera_error, size=num_photons)

    # 3) Quantize to subpixels
    x_fluo = np.round(x_with_error * subpixel) / subpixel
    y_fluo = np.round(y_with_error * subpixel) / subpixel

    return x_fluo, y_fluo


def plot_fluorophore(x_fluo, y_fluo, num_pixels=8, bg_rate=None):
    """
    Plot the simulated background events and, optionally, a single fluorophore.
    """
    plt.figure(figsize=(6, 6))

    plt.scatter(x_fluo, y_fluo, s=20, color='red', alpha=0.7,
                    label=f'Num fluorophores ({len(x_fluo)})')

    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Simulated single fluorophore')
    plt.xlim(-num_pixels / 2, num_pixels / 2)
    plt.ylim(-num_pixels / 2, num_pixels / 2)
    plt.grid(True)
    if bg_rate is not None:
        plt.legend(title=f"bg_rate_200ms_px = {bg_rate:.2f}")
    else:
        plt.legend()
    plt.show()


if __name__ == '__main__':

    # Simulate single fluorophore in the center with params defined above
    x_fluo, y_fluo = simulate_fluorophore(binding_time=s.binding_time_ms,
                                          brightness=s.brightness,
                                          sigma_psf=s.sigma_psf,
                                          camera_error=s.camera_error,
                                          subpixel=s.subpixel)

    # Print stats
    print(f"Fluorophore: {s.num_photons} photons, s_psf={s.sigma_psf}.")

    # 3) Plot them together
    plot_fluorophore(x_fluo, y_fluo, s.num_pixels)