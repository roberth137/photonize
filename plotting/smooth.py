import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Generate synthetic histogram data
np.random.seed(0)
data = np.random.normal(100, 20, 1000)
hist, bin_edges = np.histogram(data, bins=30)

# Apply the moving average filter
smoothed_hist = moving_average(hist, window_size=5)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(hist, label="Original Histogram", linestyle='--')
plt.plot(smoothed_hist, label="Smoothed Histogram (Moving Average)", color='magenta')
plt.legend()
plt.xlabel("Bin")
plt.ylabel("Frequency")
plt.title("Noise Reduction with Moving Average Filter")
plt.show()