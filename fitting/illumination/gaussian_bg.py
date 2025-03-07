import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
#from sklearn.gaussian_process.approximate import GaussianProcessRegressor

# Suppose we have arrays of localization coordinates and their measured background values
# coords: shape (N,2) array of [x_i, y_i] positions for N localizations
# background: shape (N,) array of background measurements corresponding to those localizations

# Define a kernel: RBF for smooth spatial correlation, plus a WhiteKernel for noise

def define_gaussian_model(localizations, subsampling=100):
    """
    Input:
    - dataframe of localizations with columns x,y,bg
    """
    localizations = localizations[(localizations.frame % subsampling == 0)]
    coords = np.array([localizations['x'].to_numpy(), localizations['y'].to_numpy()]).T
    print(coords)
    print(coords.shape)
    background = localizations['bg'].to_numpy()
    kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=0.5)
    #gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)  # Reduce optimization complexity

    gp.fit(coords, background)
    return gp


# Define a helper to get interpolated background at any set of points
def interpolate_background(x_query, y_query, gp):
    pts = np.column_stack([x_query, y_query])
    return gp.predict(pts)

# Example: get background at the original points (could also create a grid for a map)
#bg_pred = interpolate_background(coords[:,0], coords[:,1])


# Function to generate a 2D grid for interpolation
def create_grid(coords, grid_size=1.0):
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

    x_grid = np.arange(x_min, x_max, grid_size)
    y_grid = np.arange(y_min, y_max, grid_size)

    X, Y = np.meshgrid(x_grid, y_grid)
    return X, Y


# Example: Fit a 2D Gaussian function to the interpolated data
def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    return offset + amplitude * np.exp(-(((x - x0) ** 2 / (2 * sigma_x ** 2)) + ((y - y0) ** 2 / (2 * sigma_y ** 2))))


# Generate the background map from Gaussian Process
def plot_gp_background(coords, interpolate_background, gp, grid_size=1.0):
    X, Y = create_grid(coords, grid_size)
    Z = interpolate_background(X.ravel(), Y.ravel(), gp=gp).reshape(X.shape)

    # Fit a 2D Gaussian to the interpolated background
    xy_data = np.vstack([X.ravel(), Y.ravel()])
    initial_guess = [np.max(Z), np.mean(coords[:, 0]), np.mean(coords[:, 1]), 10.0, 10.0, np.min(Z)]

    try:
        params, _ = curve_fit(gaussian_2d, xy_data, Z.ravel(), p0=initial_guess)
        fitted_Z = gaussian_2d(xy_data, *params).reshape(X.shape)
    except RuntimeError:
        print("Gaussian fitting failed, using raw interpolated data only.")
        fitted_Z = None

    # Plot side by side: Interpolated BG map and Fitted Gaussian
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the interpolated background with localization points
    im1 = axes[0].imshow(Z.T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='viridis',
                         interpolation='nearest')
    axes[0].set_title("Interpolated Background (GP)")
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Intensity')

    # Overlay localizations in red
    axes[0].scatter(coords[:, 0], coords[:, 1], s=1, color='red', alpha=0.5, label="Localizations")
    axes[0].legend(loc="upper right")

    # Plot the fitted Gaussian (if successful)
    if fitted_Z is not None:
        im2 = axes[1].imshow(fitted_Z.T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='viridis',
                             interpolation='nearest')
        axes[1].set_title("Fitted Gaussian")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Intensity')
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load localization data
    filename = '/Users/roberthollmann/Desktop/resi-flim/data/ml/single/a565_200ms_pf.hdf5'
    localizations = pd.read_hdf(filename, key='locs')
    coords = localizations[['x', 'y']].values
    gp = define_gaussian_model(localizations)
    print('model defined')

    # Plot the background with localizations
    plot_gp_background(coords, interpolate_background, gp)