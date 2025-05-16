import simulate as s
from fitting import localization
import numpy as np
import matplotlib.pyplot as plt

def analyze_sim_event(
    x_fluo, y_fluo,
    x_bg, y_bg,
    x_entry, y_entry,
    bg_rate=s.bg_rate,
    diameter=s.fitting_diameter,
    method='com',
    sigma=None,
    binding_time=200,
    return_bg=False
):
    """
    Analyze a simulated event using COM, full‐EM MLE, or fixed‐σ fixed‐B MLE.

    Parameters
    ----------
    x_fluo, y_fluo : np.ndarray
        Signal photon coordinates.
    x_bg, y_bg : np.ndarray
        Background photon coordinates.
    x_entry, y_entry : float
        Center of ROI (initial guess).
    sigma : float
        Known PSF standard deviation (for 'mle_fixed').
    bg_rate : float
        Background photon rate (unused here, since we count actual bg in ROI).
    diameter : float
        Diameter of ROI (pixels).
    method : {'com', 'mle', 'mle_fixed'}
        'com'       → center‐of‐mass
        'mle'       → full EM fit (μ, σ, f free)
        'mle_fixed' → EM fit with σ & B fixed (μ only)

    Returns
    -------
    (x_fit, y_fit) : tuple of floats
        Fitted position, or (None, None) if no photons in ROI.
    """

    # 1) Merge all photons
    x_all = np.concatenate([x_fluo, x_bg])
    y_all = np.concatenate([y_fluo, y_bg])

    # 2) Quick ROI check (any photons at all?)
    x_roi, y_roi, _ = s.filter_points_by_radius(
        x_all, y_all,
        x_entry, y_entry,
        max_dist=diameter * 0.5
    )
    if len(x_roi) == 0:
        print("No events found within the ROI.")
        return None, None, None

    method = method.lower()
    if method == 'com':
        # center-of-mass on photons in ROI
        x_fit, y_fit, _, _ = localization.localize_com(
            x_roi, y_roi,
            return_sd=False
        )
        bg_fit = bg_rate

    elif method == 'mle_once':
        x_bg_roi, y_bg_roi, _ = s.filter_points_by_radius(
            x_bg, y_bg,
            x_entry, y_entry,
            max_dist=diameter * 0.5
        )
        fit_area = np.pi * (diameter / 2) ** 2

        B = bg_rate * fit_area * binding_time / 200
        B_2 = len(x_bg_roi)

        result = localization.mle_fixed_sigma_bg(
            x_all, y_all,
            x_start=x_entry,
            y_start=y_entry,
            diameter=diameter,
            sigma=sigma,
            bg_rate=bg_rate,
            binding_time=binding_time,
            max_iter=1

        )
        x_fit = result['mu_x']
        y_fit = result['mu_y']
        bg_fit = result['bg_rate']
        print(f"MLE fixed σ,B: iterations={result['iters']}, B_used={B}")

    elif method == 'mle':
        # full EM: fits μ, σ_x/y, and signal fraction f
        result = localization.mle_continuous(
            x_all, y_all,
            x_entry, y_entry,
            diameter
        )
        x_fit = result['mu_x']
        y_fit = result['mu_y']
        bg_fit = result['bg_rate']
        #print(f"MLE full fit: f={result['f']:.3f}")
        #print(result)


    elif method == 'mle_fixed':
        # 3) Count background photons inside ROI
        x_bg_roi, y_bg_roi, _ = s.filter_points_by_radius(
            x_bg, y_bg,
            x_entry, y_entry,
            max_dist=diameter * 0.5
        )
        fit_area = np.pi * (diameter/2)**2

        B = bg_rate * fit_area * binding_time / 200
        B_2 = len(x_bg_roi)

        print(f'bg_rate vs len(x_bg)" {B, B_2}')

        # 4) EM with σ and B fixed → only fit μ
        result = localization.mle_fixed_sigma_bg(
            x_all, y_all,
            x_start   = x_entry,
            y_start   = y_entry,
            diameter  = diameter,
            sigma     = sigma,
            bg_rate = bg_rate,
            binding_time = binding_time
        )
        x_fit = result['mu_x']
        y_fit = result['mu_y']
        bg_fit = result['bg_rate']
        print(f"MLE fixed σ,B: iterations={result['iters']}, B_used={B}")

    elif method == 'pass':
        x_fit, y_fit, bg_fit = x_entry, y_entry, bg_rate

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'pass', 'com', 'mle', or 'mle_fixed'.")

    return x_fit, y_fit, bg_fit



def plot_analysis(x_fluo, y_fluo, x_bg, y_bg,
                  x_ref=s.x_ref, y_ref=s.y_ref,
                  diameter=s.fitting_diameter,
                  method='com',
                  num_pixels=s.num_pixels,
                  sigma=None):
    """
    Plot the background events (blue) together with the simulated fluorophore (red)
    in the same coordinate space.
    """
    plt.figure(figsize=(6, 6))

    x_fit, y_fit = analyze_sim_event(x_fluo, y_fluo,
                                     x_bg, y_bg,
                                     x_ref, y_ref,
                                     diameter=diameter,
                                     method=method,
                                     sigma=sigma)

    all_x = np.append(x_fluo, x_bg)
    all_y = np.append(y_fluo, y_bg)

    x_all_cons, y_all_cons, _ = s.filter_points_by_radius(all_x, all_y, 0, 0, max_dist=(diameter/2))
    x_fluo_cons, y_fluo_cons, _ = s.filter_points_by_radius(x_fluo, y_fluo, 0, 0, max_dist=(diameter/2))
    x_bg_cons, y_bg_cons, _ = s.filter_points_by_radius(x_bg, y_bg, 0, 0, max_dist=(diameter/2))

    # Plot background events
    plt.scatter(x_bg, y_bg, s=10, color='blue', alpha=0.4, label='bg photons')
    # Plot fluorophore photons
    plt.scatter(x_fluo, y_fluo, s=10, color='red', alpha=0.4,
                label=f'({s.num_photons} signal photons, σ={s.sigma_psf})')
    plt.scatter(x_all_cons, y_all_cons, s=10, color='red', alpha=1,
                label=f'Fitting ({len(x_all_cons)}) photons, {len(x_fluo_cons)} signal, {len(x_bg_cons)} bg')
    plt.scatter(x_bg_cons, y_bg_cons, s=10, color='blue', alpha=1,
                label=f'{len(x_bg_cons)} bg photons considered')
    plt.scatter(x_fit, y_fit, marker='x',  s=50, color='purple', alpha=0.7,
                label=f'Fitting: {method}, Pos: ({x_fit:.4f}|{y_fit:.4f})')

    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Analyze Background + Single Fluorophore')
    plt.xlim(-num_pixels / 2, num_pixels / 2)
    plt.ylim(-num_pixels / 2, num_pixels / 2)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    analysis_method = 'com' #valid are 'com', 'mle', 'mle_fixed'
    diameter = 5

    # Simulate single fluorophore
    x_fluo, y_fluo = s.simulate_fluorophore(int(s.num_photons), sigma_psf=s.sigma_psf)
    # Simulate background
    x_bg, y_bg = s.simulate_background(s.num_pixels, s.binding_time_ms,
                                       s.bg_rate, s.subpixel)

    # Plot both together
    plot_analysis(x_fluo, y_fluo, x_bg, y_bg,
                  x_ref=0, y_ref=0,
                  method=analysis_method,
                  diameter=diameter,
                  sigma=s.sigma_psf)
