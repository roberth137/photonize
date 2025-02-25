import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Parameters
# -----------------------------
mean_photons = 250  # Target mean photon count
min_photons = 100  # Minimum photon count
size = 1000  # Number of fluorophores/events

# Time windows (in ns) to consider
time_windows = np.arange(1,35)#[5, 10, 15, 20, 25, 30, 35]

# Fluorophore lifetimes (in ns)
lifetimes = [3.6, 4.1]

# -----------------------------
# 1. Photon Count Simulation
# -----------------------------
# We want a shifted exponential distribution:
#   mean = (mean_photons - min_photons)
# => lambda = 1 / (mean_photons - min_photons)
decay_binding = 1 / (mean_photons - min_photons)

# Generate floating-point photon counts from exponential
photons_float = np.random.exponential(scale=1 / decay_binding, size=size) + min_photons

# Round to integers
photons = np.round(photons_float).astype(int)

# -----------------------------
# 2. Arrival Time Simulation
# -----------------------------
# For each lifetime, we will:
#   - For each event i, generate 'photons[i]' arrival times
#   - Collect all arrival times for that lifetime
#   - For each time window, compute the average of arrival_times % time_window

avg_arrival_dict = {}  # { lifetime: [avg_for_tw_1, avg_for_tw_2, ...], ... }

for lifetime in lifetimes:
    # Collect all photon arrival times (across all events) for this lifetime
    all_arrivals = []

    # Generate arrival times event by event
    for i in range(size):
        # For event i, we have photons[i] photons
        # Each photon arrival time ~ Exp(lifetime)
        times_i = np.random.exponential(scale=lifetime, size=photons[i])
        all_arrivals.append(times_i)

    # Flatten into a single array of arrival times for this lifetime
    all_arrivals = np.concatenate(all_arrivals)

    # Now compute average arrival times for each time window
    avg_arrival_times = []
    for tw in time_windows:
        mod_times = all_arrivals % tw
        avg_arrival_times.append(np.mean(mod_times))

    avg_arrival_dict[lifetime] = avg_arrival_times

# -----------------------------
# 3. Plotting
# -----------------------------
plt.figure(figsize=(14, 6))

# (A) Left: Photon Count Distribution
plt.subplot(1, 2, 1)
plt.hist(photons, bins=range(min(photons), max(photons) + 1),
         density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel("Photon Count")
plt.ylabel("Probability Density")
plt.title("Simulated Photon Count Distribution")

# (B) Right: Average Arrival Time vs. Time Window
plt.subplot(1, 2, 2)
for lifetime in lifetimes:
    plt.plot(time_windows,
             avg_arrival_dict[lifetime],
             marker='o',
             label=f"Lifetime {lifetime} ns")

plt.xlabel("Time Window (ns)")
plt.ylabel("Average Arrival Time (ns)")
plt.title("Average Arrival Time vs. Time Window")
plt.legend()

plt.tight_layout()
plt.show()