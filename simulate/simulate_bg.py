import numpy as np
import matplotlib.pyplot as plt


def simulate_background_single_value(num_pixels=8, binding_time_ms=200,
                                     bg_200ms_px=6, subpixel_levels=16):
    """
    Simulate background events over an 8x8 pixel area using a single random
    background value for all pixels, returning arrays of x_coords, y_coords.

    Coordinates are mapped into [-4,4) along x and y.
    """
    # Total area = num_pixels * num_pixels (64 for 8x8)
    total_area = num_pixels * num_pixels

    # Compute the expected number of background events, scaling by binding time
    expected_counts = total_area * bg_200ms_px * (binding_time_ms / 200.0)

    # Draw the total number of background events from a Poisson distribution
    num_events = np.random.poisson(expected_counts)

    # Randomly assign subpixel coordinates in [0, num_pixels), then shift to [-4,4)
    x_coords = (np.random.randint(0, num_pixels * subpixel_levels, size=num_events)
                / subpixel_levels - num_pixels / 2)
    y_coords = (np.random.randint(0, num_pixels * subpixel_levels, size=num_events)
                / subpixel_levels - num_pixels / 2)

    return x_coords, y_coords


def plot_background(x_coords, y_coords, num_pixels=8, bg_rate=None):
    """
    Plot the simulated background events.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, s=20, color='blue', alpha=0.7, label='Background events')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Simulated Background Events (8×8 pixels)')
    plt.xlim(-num_pixels/2, num_pixels/2)
    plt.ylim(-num_pixels/2, num_pixels/2)
    plt.grid(True)
    if bg_rate is not None:
        plt.legend(title=f"bg_rate = {bg_rate:.2f}")
    else:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    # Simulation parameters
    num_pixels = 8  # Area: 8x8 pixels
    binding_time_ms = 1000  # Binding time in ms
    bg_mean = 6  # Mean background per pixel (for 200 ms)
    bg_std = 1  # Standard deviation of background per pixel (for 200 ms)
    subpixel_levels = 16  # Subpixel resolution: each pixel is divided into 16 parts

    # Simulate background events
    x_coords, y_coords = simulate_background_single_value(num_pixels,
                                                                   binding_time_ms,
                                                                   bg_mean,
                                                                   subpixel_levels)
    print(f"Simulated with a background rate of: {bg_mean:.2f} counts per pixel (for 200 ms)")

    # Plot the simulated events
    plot_background(x_coords, y_coords, num_pixels, bg_mean)