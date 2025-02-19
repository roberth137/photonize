import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Parameters
# -----------------------------
mean_photons = 250  # Target mean photon count
min_photons = 100   # Minimum photon count
size = 1000000         # Number of simulated events (fluorophores)

# Time windows (in ns) to consider
time_windows = [5, 10, 15, 20, 30, 35]

# Fluorophore lifetimes (in ns)
lifetimes = [3.6, 4.1]

# -----------------------------
# Photon Count Simulation
# -----------------------------
# For a shifted exponential distribution:
# Standard exponential mean: 1/λ, so we want (1/λ) = (mean_photons - min_photons)
decay_binding = 1 / (mean_photons - min_photons)
# Generate photon counts and shift them
photons_float = np.random.exponential(scale=1/decay_binding, size=size) + min_photons
photons = np.round(photons_float).astype(int)

# -----------------------------
# Average Arrival Times Simulation
# -----------------------------
# We'll simulate arrival times for each fluorophore (for each lifetime)
# and then compute the average arrival time within each specified time window.
avg_arrival_dict = {}  # Dictionary to store average arrival times

for lifetime in lifetimes:
    avg_arrival_times = []
    # For each time window, simulate arrival times and compute average modulo that window
    for tw in time_windows:
        arrival_times = np.random.exponential(scale=lifetime, size=size)
        arrival_times_in_window = arrival_times % tw
        avg_arrival_times.append(np.mean(arrival_times_in_window))
    avg_arrival_dict[lifetime] = avg_arrival_times

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(14, 6))

# Graph 1: Photon Count Distribution Histogram
plt.subplot(1, 2, 1)
plt.hist(photons, bins=range(min(photons), max(photons) + 1), density=True,
         alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel("Photon Count")
plt.ylabel("Probability Density")
plt.title("Simulated Photon Count Distribution")

# Graph 2: Average Arrival Time vs. Time Window for each Lifetime
plt.subplot(1, 2, 2)
for lifetime in lifetimes:
    plt.plot(time_windows, avg_arrival_dict[lifetime], marker='o', label=f"Lifetime {lifetime} ns")
plt.xlabel("Time Window (ns)")
plt.ylabel("Average Arrival Time (ns)")
plt.title("Average Arrival Time vs. Time Window")
plt.legend()

plt.tight_layout()
plt.show()