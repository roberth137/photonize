import simulate as s
from fitting import localization
import numpy as np
import matplotlib.pyplot as plt


def analyze_sim_event(x_fluo, y_fluo,
                      x_bg, y_bg,
                      x_entry, y_entry,
                      bg_rate=s.bg_rate,
                      diameter=s.fitting_diameter,
                      method='com',
                      consider_bg=False):
    """
    Analyze a simulated event using COM or MLE fitting.

    Parameters
    ----------
    x_fluo, y_fluo : np.ndarray
        Signal photon coordinates.
    x_bg, y_bg : np.ndarray
        Background photon coordinates.
    x_entry, y_entry : float
        Center of ROI.
    bg_rate : float
        Background photon rate (for MLE).
    diameter : float
        Diameter of ROI (pixels).
    consider_bg : bool
        Whether to account for background photons.
    method : {'com', 'mle'}
        Localization method to use.

    Returns
    -------
    x_fit, y_fit : tuple of floats
        Fitted position, or None if no events found.
    """
    import numpy as np

    # Merge signal and background
    x_all = np.concatenate([x_fluo, x_bg])
    y_all = np.concatenate([y_fluo, y_bg])

    # Crop to circular ROI
    x_roi, y_roi, _ = filter_points_by_radius(x_all, y_all, x_entry, y_entry, max_dist=diameter / 2.0)
    if len(x_roi) == 0:
        print("No events found within the ROI.")
        return None

    method = method.lower()
    if method == 'com':
        #bg_count = bg_rate * (s.binding_time_ms / 200) * s.fit_area if consider_bg else None
        x_fit, y_fit = localization.localize_com(x_roi, y_roi, return_sd=False)

    elif method == 'mle':
        sigma_psf = (s.sx + s.sy) / 2.0
        x_fit, y_fit = localization.mle_position(x_roi, y_roi,
                                                 x_entry, y_entry,
                                                 bg_rate=bg_rate if consider_bg else 0,
                                                 sigma_psf=sigma_psf,
                                                 diameter=diameter)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'com' or 'mle'.")

    return x_fit, y_fit


def plot_analysis(x_fluo, y_fluo, x_bg, y_bg,
                  x_ref=s.x_ref, y_ref=s.y_ref,
                  diameter=s.fitting_diameter,
                  num_pixels=s.num_pixels):
    """
    Plot the background events (blue) together with the simulated fluorophore (red)
    in the same coordinate space.
    """
    plt.figure(figsize=(6, 6))

    x_fit, y_fit = analyze_sim_event(x_fluo, y_fluo,
                                     x_bg, y_bg,
                                     x_ref, y_ref,
                                     diameter,
                                     consider_bg=False)
    x_fit_w_bg, y_fit_w_bg = analyze_sim_event(x_fluo, y_fluo,
                                               x_bg, y_bg,
                                               x_ref, y_ref,
                                               diameter,
                                               consider_bg=True)

    all_x = np.append(x_fluo, x_bg)
    all_y = np.append(y_fluo, y_bg)

    x_cons, y_cons, _ = s.filter_points_by_radius(all_x, all_y, 0, 0, max_dist=s.max_dist)
    x_bg_cons, y_bg_cons, _ = s.filter_points_by_radius(x_bg, y_bg, 0, 0, max_dist=s.max_dist)

    # Plot background events
    plt.scatter(x_bg, y_bg, s=10, color='blue', alpha=0.4, label='bg photons')
    # Plot fluorophore photons
    plt.scatter(x_fluo, y_fluo, s=10, color='red', alpha=0.4,
                label=f'({s.num_photons} signal photons, σ={s.sigma_psf})')
    plt.scatter(x_cons, y_cons, s=10, color='red', alpha=1,
                label=f'Considered for fitting: ({len(x_cons)} photons')
    plt.scatter(x_bg_cons, y_bg_cons, s=10, color='blue', alpha=1,
                label=f'{len(x_bg_cons)} bg photons considered')
    plt.scatter(
        x_fit,
        y_fit,
        marker='x',  # Use 'x' to draw a cross
        s=50,
        color='purple',
        alpha=0.7,
        label=f'Plain COM (x|y): ({x_fit:.4f}|{y_fit:.4f})'
    )
    plt.scatter(
        x_fit,
        y_fit,
        marker='x',  # Use 'x' to draw a cross
        s=50,
        color='purple',
        alpha=0.7,
        label=f'COM cons. bg (x|y): ({x_fit_w_bg:.4f}|{y_fit_w_bg:.4f})'
    )

    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Analyze Background + Single Fluorophore')
    plt.xlim(-num_pixels / 2, num_pixels / 2)
    plt.ylim(-num_pixels / 2, num_pixels / 2)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Simulate a single fluorophore event
    x_fluo, y_fluo = s.simulate_fluorophore(int(s.num_photons), sigma_psf=s.sigma_psf)

    # Simulate background events
    x_bg, y_bg = s.simulate_background(s.num_pixels, s.binding_time_ms,
                                       s.bg_rate, s.subpixel)

    # Plot both together
    plot_analysis(x_fluo, y_fluo, x_bg, y_bg, x_ref=0, y_ref=0, diameter=s.fitting_diameter)
