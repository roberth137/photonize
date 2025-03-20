import matplotlib.pyplot as plt
import numpy as np

from simulate.simulate_event_statistics import simulate_filtered_events
from simulate.simulate_fluorophore import simulate_fluorophore
from simulate.simulate_bg import simulate_background_single_value
from fitting import event_position


def plot_combined(x_fluo, y_fluo, x_bg, y_bg, num_pixels=8, bg_rate=None,
                  num_photons=500, sigma=1.1):
    """
    Plot the background events (blue) together with the simulated fluorophore (red)
    in the same coordinate space.
    """
    plt.figure(figsize=(6, 6))

    all_x = np.append(x_fluo, x_bg)
    all_y = np.append(y_fluo, y_bg)

    print(all_x)

    x_fit, y_fit, sdx, sdy = event_position(all_x, all_y, return_sd=False)

    # Plot background events
    plt.scatter(x_bg, y_bg, s=10, color='blue', alpha=0.7, label='Background events')
    # Plot fluorophore photons
    plt.scatter(x_fluo, y_fluo, s=10, color='red', alpha=0.7,
                label=f'Fluorophore ({num_photons} photons, σ={sigma})')

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
    if bg_rate is not None:
        plt.legend(title=f"bg_200ms_px = {bg_rate:.2f}")
    else:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    # Simulation parameters
    num_pixels = 8  # 8x8 area; coordinates range from -4 to 4.
    binding_time_ms = 300  # Actual binding time in ms (scaling from the 200 ms reference)
    brightness = 1
    bg_200ms_px = 1  # Mean background events per pixel for 200 ms
    subpixel_levels = 16  # Subpixel resolution
    num_photons = binding_time_ms * brightness  # Total number of photons for the fluorophore event
    sigma = 1.1  # Standard deviation of the fluorophore (in pixels)

    # Simulate a single fluorophore event
    x_fluo, y_fluo = simulate_fluorophore(num_photons, sigma)

    # Simulate background events
    x_bg, y_bg = simulate_background_single_value(num_pixels, binding_time_ms,
                                                  bg_200ms_px, subpixel_levels)


    # Plot both together
    plot_combined(x_fluo, y_fluo, x_bg, y_bg, num_pixels,
                  bg_rate=bg_200ms_px, num_photons=num_photons, sigma=sigma)