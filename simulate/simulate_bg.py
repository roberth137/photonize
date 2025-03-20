import numpy as np
import matplotlib.pyplot as plt
import simulate as s

def simulate_background(num_pixels=s.num_pixels,
                        binding_time_ms=s.binding_time_ms,
                        bg_rate=s.bg_rate,
                        min_cam_binning=s.min_cam_binning):
    """
    Simulate background events over an 8x8 pixel area using a single random
    background value for all pixels, returning arrays of x_coords, y_coords.

    Coordinates are mapped into [-4,4) along x and y.
    """
    # Total area = num_pixels * num_pixels (64 for 8x8)
    total_area = num_pixels * num_pixels

    # Compute the expected number of background events, scaling by binding time
    expected_counts = total_area * bg_rate * (binding_time_ms / 200.0)

    # Draw the total number of background events from a Poisson distribution
    num_events = np.random.poisson(expected_counts)

    # Randomly assign subpixel coordinates in [0, num_pixels), then shift to [-4,4)
    x_coords = (np.random.randint(0, num_pixels / min_cam_binning, size=num_events)
                * min_cam_binning - num_pixels / 2)
    y_coords = (np.random.randint(0, num_pixels / min_cam_binning, size=num_events)
                * min_cam_binning - num_pixels / 2)

    return x_coords, y_coords


def plot_background(x_coords, y_coords,
                    num_pixels=s.num_pixels, bg_rate=s.bg_rate):
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
    plt.legend(title=f"bg_rate = {s.bg_rate:.2f}")
    plt.show()


if __name__ == '__main__':
    # Simulate background events
    x_coords, y_coords = simulate_background(num_pixels=s.num_pixels,
                                            binding_time_ms=s.binding_time_ms,
                                            bg_rate=s.bg_rate,
                                            min_cam_binning=s.min_cam_binning)
    print(f"Simulated with a background rate of: {s.bg_rate:.2f} counts per pixel (for 200 ms)")

    # Plot the simulated events
    plot_background(x_coords, y_coords, s.num_pixels, s.bg_rate)