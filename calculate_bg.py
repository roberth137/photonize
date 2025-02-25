from math import floor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_bg_map(localizations):
    """
    Create a 2D map of average background values from localization data.

    Parameters
    ----------
    localizations

    Returns
    -------
    bg_map : 2D numpy.ndarray
        A 2D array of average background values per pixel.
        Shape is determined by the max of the computed pixel indices.
    """
    x = np.asarray(localizations.x, dtype=np.float64)
    y = np.asarray(localizations.y, dtype=np.float64)
    bg = np.asarray(localizations.bg, dtype=np.float64)

    px_x = np.round(x).astype(int)
    px_y = np.round(y).astype(int)

    # Determine the shape of the output matrix
    max_x = px_x.max()
    max_y = px_y.max()
    # If indices start at 0, the size is max_index + 1
    bg_map = np.zeros((max_x + 1, max_y + 1), dtype=np.float64)
    count_map = np.zeros((max_x + 1, max_y + 1), dtype=np.int64)

    # Accumulate sums of background and counts using a fast NumPy approach
    np.add.at(bg_map, (px_x, px_y), bg)
    np.add.at(count_map, (px_x, px_y), 1)

    # Convert summed values to averages where count > 0
    mask = (count_map > 0)
    bg_map[mask] /= count_map[mask]

    return bg_map


# Example usage:
if __name__ == "__main__":
    filename = 't/orig58_all_f.hdf5'
    localizations = pd.read_hdf(filename, key='locs')

    # Generate synthetic data
    np.random.seed(0)
    n_points = 10_000

    # Suppose your localizations are in the range [0, 200) in both x and y
    x_data = np.random.rand(n_points) * 200.0
    y_data = np.random.rand(n_points) * 200.0

    # Random background intensities
    bg_data = np.random.rand(n_points) * 100.0

    # Compute background map using 1.0 pixel size and floor indexing
    bg_map = compute_bg_map(localizations)

    print("bg_map shape:", bg_map.shape)
    print("A small slice of bg_map:\n", bg_map[:5, :5])

    plt.figure(figsize=(6, 5))

    # imshow expects images in (row, col) format, which here is (x, y).
    # By default, the origin of the matrix is at the top-left corner.
    # You can flip or specify extent/origin to preserve spatial orientation.
    plt.imshow(bg_map.T,
               origin='lower',  # Make (0,0) appear at bottom-left
               cmap='viridis',  # Colormap choice
               interpolation='nearest',
               aspect='auto')

    plt.colorbar(label="Background intensity")
    plt.title("Background Map")
    plt.xlabel("X pixel index")
    plt.ylabel("Y pixel index")
    plt.show()

