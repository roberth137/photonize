import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

# Simulating data similar to the histogram
np.random.seed(0)
data = np.concatenate([np.random.poisson(2, 100),
                       np.random.poisson(10, 100),
                       np.random.poisson(2, 100)])

# Fit a step function using change point detection
model = "l2"  # Least squares cost function
algo = rpt.Binseg(model=model).fit(data)
change_points = algo.predict(n_bkps=2)  # Detect 2 change points (for on and off)

# Plotting the result
plt.figure(figsize=(10, 6))
plt.bar(range(len(data)), data, color='blue', alpha=0.6, label="Photon Counts")
for cp in change_points[:-1]:  # Ignore the last point as it's the end
    plt.axvline(cp, color='red', lw=2)
plt.title("Step Function Fit to Photon Histogram")
plt.xlabel("Time Bins")
plt.ylabel("Photon Counts")
plt.legend()
plt.show()