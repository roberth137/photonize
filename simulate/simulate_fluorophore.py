import numpy as np
import matplotlib.pyplot as plt


def simulate_fluorophore(num_photons=500, sigma=1.1):
    """
    Simulate one fluorophore at the origin (0,0),
    with photon positions sampled from a 2D Gaussian.

    num_photons (int): total number of photons to emit.
    sigma (float): standard deviation of the 2D Gaussian in pixel units.

    Returns:
        x_fluo (np.ndarray): x-coordinates of emitted photons.
        y_fluo (np.ndarray): y-coordinates of emitted photons.
    """
    x_fluo = np.random.normal(loc=0.0, scale=sigma, size=num_photons)
    y_fluo = np.random.normal(loc=0.0, scale=sigma, size=num_photons)
    return x_fluo, y_fluo


def plot_fluorophore(x_fluo, y_fluo, num_pixels=8, bg_rate=None):
    """
    Plot the simulated background events and, optionally, a single fluorophore.
    """
    plt.figure(figsize=(6, 6))

    plt.scatter(x_fluo, y_fluo, s=20, color='red', alpha=0.7,
                    label='Fluorophore (500 photons)')

    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.title('Simulated Background + Single Fluorophore')
    plt.xlim(-num_pixels / 2, num_pixels / 2)
    plt.ylim(-num_pixels / 2, num_pixels / 2)
    plt.grid(True)
    if bg_rate is not None:
        plt.legend(title=f"bg_rate_200ms_px = {bg_rate:.2f}")
    else:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    # Simulation parameters
    num_pixels = 8  # 8x8 area => coordinates range from -4 to +4
    binding_time_ms = 1000  # Binding time in ms (scaled from a 200 ms reference)
    subpixel_levels = 16  # Subpixel resolution
    num_photons_fluorophore = 500
    sigma_fluorophore = 1.1  # Gaussian width (standard deviation)

    # 2) Simulate single fluorophore in the center with 500 photons at sigma=1.1
    x_fluo, y_fluo = simulate_fluorophore(num_photons=num_photons_fluorophore,
                                          sigma=sigma_fluorophore)

    # Print stats
    print(f"Fluorophore: {num_photons_fluorophore} photons, Gaussian sigma={sigma_fluorophore}.")

    # 3) Plot them together
    plot_fluorophore(x_fluo, y_fluo, num_pixels)