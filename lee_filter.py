import numpy as np
import matplotlib.pyplot as plt


def lee_filter_1d(signal, window_size=5):
    """Applies a Lee filter to a 1D signal."""
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2

    for i in range(half_window, len(signal) - half_window):
        local_region = signal[i - half_window: i + half_window + 1]
        local_mean = np.mean(local_region)
        local_variance = np.var(local_region)
        signal_variance = np.var(signal)  # Estimated global noise

        # Compute the smoothing coefficient k
        k = local_variance / (local_variance + signal_variance)

        # Apply the filter
        filtered_signal[i] = signal[i] + k * (local_mean - signal[i])

    # Edge handling (copy original signal for the edges)
    filtered_signal[:half_window] = signal[:half_window]
    filtered_signal[-half_window:] = signal[-half_window:]

    return filtered_signal


# Generate synthetic noisy signal for demonstration
np.random.seed(42)
time = np.linspace(0, 10, 1000)
true_signal = np.sin(time * 2 * np.pi / 5)
noise = np.random.normal(0, 0.2, size=len(true_signal))
noisy_signal = true_signal + noise

# Apply the Lee filter
filtered_signal = lee_filter_1d(noisy_signal, 2)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time, noisy_signal, label='Noisy Signal', alpha=0.5)
plt.plot(time, filtered_signal, label='Lee Filtered Signal', color='red')
plt.plot(time, true_signal, label='True Signal', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.title('Lee Filter Applied to a Noisy Single Molecule Signal')
plt.legend()
plt.show()