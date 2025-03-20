import simulate as s
from fitting import event_position
import numpy as np
import matplotlib.pyplot as plt

def analyze_event(x_fluo, y_fluo, 
                  x_bg, y_bg, 
                  x_entry, y_entry,
                  diameter,
                  consider_bg=False):
    """
    takes signal and background arrays and performs center of mass fit with event_position
    takes circular roi around x_entry and y_entry
    """
def distance_to_point(x, y, x_ref, y_ref, max_dist=s.max_dist):
    """
    Calculate the distance from each (x[i], y[i]) to the given point.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    point : tuple or list of length 2
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


def plot_analysis(x_fluo, y_fluo, x_bg, y_bg, num_pixels=s.num_pixels):
    """
    Plot the background events (blue) together with the simulated fluorophore (red)
    in the same coordinate space.
    """
    plt.figure(figsize=(6, 6))

    all_x = np.append(x_fluo, x_bg)
    all_y = np.append(y_fluo, y_bg)

    print(all_x)

    x_considered, y_considered, _ = s.distance_to_point(all_x,
                                                        all_y,
                                                        0, 0, max_dist=s.max_dist)

    x_fit, y_fit, sdx, sdy = event_position(x_considered, y_considered, return_sd=True)

    # Plot background events
    plt.scatter(x_bg, y_bg, s=10, color='blue', alpha=0.7, label='Background events')
    # Plot fluorophore photons
    plt.scatter(x_fluo, y_fluo, s=10, color='red', alpha=0.4,
                label=f'Fluorophore ({s.num_photons} photons, σ={s.sigma_psf})')
    plt.scatter(x_considered, y_considered, s=10, color='green', alpha=1,
                label=f'Considered for fitting: ({len(x_considered)} photons')

    plt.scatter(
        x_fit,
        y_fit,
        marker='x',  # Use 'x' to draw a cross
        s=50,
        color='green',
        alpha=0.7,
        label=f'Fitted position (x|y): ({x_fit:.4f}|{y_fit:.4f})'
    )

    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Simulated Background + Single Fluorophore')
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
                                                  s.bg_rate, s.min_cam_binning)


    # Plot both together
    plot_analysis(x_fluo, y_fluo, x_bg, y_bg)