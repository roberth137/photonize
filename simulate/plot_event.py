import matplotlib.pyplot as plt
import numpy as np
from fitting import event_position
import simulate as s


def plot_combined(x_fluo, y_fluo, x_bg, y_bg, num_pixels=s.num_pixels):
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
    plot_combined(x_fluo, y_fluo, x_bg, y_bg)