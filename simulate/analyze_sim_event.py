import simulate as s
from fitting import localization
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def analyze_sim_event(x_fluo, y_fluo,
                  x_bg, y_bg, 
                  x_entry, y_entry,
                  bg_rate=s.bg_rate,
                  diameter = s.fitting_diameter,
                  consider_bg=False):
    """
    Takes signal and background arrays and performs a center-of-mass fit using event_position.
    A circular region of interest (ROI) around (x_entry, y_entry) with the given diameter is used.

    Parameters
    ----------
    x_fluo, y_fluo : np.ndarray
        Arrays of x and y coordinates for the fluorophore (signal) events.
    x_bg, y_bg : np.ndarray
        Arrays of x and y coordinates for the background events.
    x_entry, y_entry : float
        Coordinates for the center of the ROI.
    diameter : float
        The diameter of the ROI (a circle).
    consider_bg : bool, optional
        If True, include background events in the analysis. Otherwise, only fluorophore events are considered.
        Default is False.

    Returns
    -------
    x_fit, y_fit, sdx, sdy : tuple of floats
        The fitted event position and its uncertainties. Returns None if no events are found in the ROI.
    """
    import numpy as np  # Ensure numpy is imported

    bg_count = bg_rate * (s.binding_time_ms/200) * s.fit_area

    x_all = np.concatenate([x_fluo, x_bg])
    y_all = np.concatenate([y_fluo, y_bg])

    # Select events within the ROI using the distance_to_point function
    x_roi, y_roi, _ = distance_to_point(x_all, y_all, x_entry, y_entry, max_dist=diameter / 2.0)

    # Check if there are any events in the ROI
    if len(x_roi) == 0:
        print("No events found within the ROI.")
        return None
    if consider_bg:
        x_fit, y_fit = localization.com_position(x_roi, y_roi,
                                                 x_entry, y_entry,
                                                 bg_count)
    else:
        x_fit, y_fit = localization.com_position(x_roi, y_roi,
                                                 x_entry, y_entry,
                                                 bg_count=None)

    return x_fit, y_fit


def distance_to_point(x, y, x_ref=s.x_ref, y_ref=s.y_ref, max_dist=None):
    """
    Calculate the distance from each (x[i], y[i]) to the given point.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    x_ref, y_ref
        The reference point (x0, y0).
    max_dist : max_dist to point

    Returns
    -------
    x: np.ndarray
    y: np.ndarray

    dist: 1D array of distances from each (x[i], y[i]) to 'point'.
    """

    dist = np.sqrt((x - x_ref)**2 + (y - y_ref)**2)

    if max_dist:
        mask = dist < max_dist
        return x[mask], y[mask], dist[mask]
    else:
        return x, y, dist


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

    x_cons, y_cons, _ = s.distance_to_point(all_x, all_y, 0, 0, max_dist=s.max_dist)
    x_bg_cons, y_bg_cons, _ = s.distance_to_point(x_bg, y_bg, 0, 0, max_dist=s.max_dist)

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