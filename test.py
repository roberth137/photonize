import numpy as np
import matplotlib.pyplot as plt

# Given parameters
mean_desired = 250  # Target mean
min_value = 100      # Minimum value

# Compute lambda for standard exponential (before shifting)
lambda_param = 1 / (mean_desired - min_value)

# Generate samples
size = 1000
samples = np.random.exponential(scale=1/lambda_param, size=size) + min_value

# Plot histogram
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b')

# Overlay theoretical PDF
x = np.linspace(min_value, max(samples), 100)
pdf = lambda_param * np.exp(-lambda_param * (x - min_value))  # Adjusted for shift
plt.plot(x, pdf, 'r-', label='Theoretical PDF')

plt.xlabel("Value")
plt.ylabel("Density")
plt.title(f"Shifted Exponential Distribution (Min = {min_value}, Mean = {mean_desired})")
plt.legend()
plt.show()