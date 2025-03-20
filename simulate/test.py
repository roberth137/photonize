import numpy as np
import matplotlib.pyplot as plt

N = 10_000  # number of samples
mu = -0.5   # mean of log(X)
sigma = 0.7 # std of log(X)

# Simulate lognormal data
data_lognormal = np.random.lognormal(mean=mu, sigma=sigma, size=N)

plt.hist(data_lognormal, bins=150, density=False, color='tomato', alpha=0.8)
plt.xlabel('Value')
plt.ylabel('Counts')
plt.title('Lognormal Distribution')
plt.show()
