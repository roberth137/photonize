import matplotlib.pyplot as plt
import numpy as np
import simulate as s


def plot_event(x_fluo, y_fluo, x_bg, y_bg, num_pixels=s.num_pixels):
    """
    Plot the background events (blue) together with the simulated fluorophore (red)
    in the same coordinate space.
    """
    plt.figure(figsize=(6, 6))

    all_x = np.append(x_fluo, x_bg)
    all_y = np.append(y_fluo, y_bg)
    x_mean, y_mean = np.mean(all_x), np.mean(all_y)

    # Plot background events
    plt.scatter(x_bg, y_bg, s=10, color='blue', alpha=0.7, label='Background events')
    # Plot fluorophore photons
    plt.scatter(x_fluo, y_fluo, s=10, color='red', alpha=1,
                label=f'Fluorophore ({s.num_photons} photons, sigma_psf={s.sigma_psf})')
    plt.scatter(x_mean, y_mean, s=20, color='green', alpha=1,
                label=f'(x|y): ({x_mean:.4f}|{y_mean:.4f})')

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
                                                  s.bg_rate_true, s.min_cam_binning)


    # Plot both together
    plot_event(x_fluo, y_fluo, x_bg, y_bg)