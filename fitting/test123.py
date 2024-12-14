import numpy as np

# Simulate data
N = 1000

# True signal: Gaussian distribution centered at 50
signal = np.random.normal(loc=50, scale=10, size=N)

# Add random Gaussian noise with mean=0 and std=5
noise = np.random.normal(loc=0, scale=5, size=N)

# Total data (signal + noise)
data_with_noise = signal + noise

# Calculate standard deviations
std_combined = np.std(data_with_noise)  # Combined data
std_noise = np.std(noise)               # Known noise
std_corrected = np.sqrt(std_combined**2 - std_noise**2)

# Print results
print(f"Combined Standard Deviation: {std_combined:.2f}")
print(f"Noise Standard Deviation: {std_noise:.2f}")
print(f"Corrected Standard Deviation (Signal Only): {std_corrected:.2f}")
