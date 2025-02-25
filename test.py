import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation Parameters
# -----------------------------
mean_photons = 450    # Target mean photon count
min_photons = 200     # Minimum photon count
size = 10000           # Number of events (fluorophores)

# Time windows (in ns) to consider
time_windows = [5, 10, 15, 20, 25, 30, 35]

# Fluorophore lifetimes (in ns)
lifetimes = [3.6, 4.1]

# Optional: set a random seed for reproducibility
# np.random.seed(42)

# -----------------------------
# 1. Photon Count Simulation
# -----------------------------
# We simulate photon counts using a shifted exponential distribution:
# The standard exponential mean should be: mean_photons - min_photons.
decay_binding = 1 / (mean_photons - min_photons)
photons_float = np.random.exponential(scale=1/decay_binding, size=size) + min_photons
photons = np.round(photons_float).astype(int)

# -----------------------------
# 2. Per-Event Average Lifetime Calculation
# -----------------------------
# For each lifetime, for each event, and for each time window,
# we will compute the per-event average arrival time (modulo the time window).
# We'll store these in a nested dictionary:
# results[lifetime][tw] will be an array of per-event averages.
results = {lifetime: {} for lifetime in lifetimes}

for lifetime in lifetimes:
    for tw in time_windows:
        event_avg_list = []  # List to store per-event average modded arrival times
        for i in range(size):
            # Generate arrival times for event i using its photon count.
            # Each arrival time is drawn from an exponential with the given lifetime.
            times = np.random.exponential(scale=lifetime, size=photons[i])
            # "Fold" the times into the time window using modulo
            mod_times = times % tw
            # Compute the average for this event (if there are photons)
            event_avg = np.mean(mod_times) if len(mod_times) > 0 else 0
            event_avg_list.append(event_avg)
        # Store the array (converted to a NumPy array for convenience)
        results[lifetime][tw] = np.array(event_avg_list)

# -----------------------------
# 3. Compute Statistics Across Events
# -----------------------------
# For each lifetime and time window, compute:
#   - mean of per-event average lifetime
#   - standard deviation (std) of per-event average lifetime
stats = {lifetime: {'avg': [], 'std': []} for lifetime in lifetimes}

for lifetime in lifetimes:
    for tw in time_windows:
        event_avgs = results[lifetime][tw]
        stats[lifetime]['avg'].append(np.mean(event_avgs))
        stats[lifetime]['std'].append(np.std(event_avgs))
    # Convert lists to arrays
    stats[lifetime]['avg'] = np.array(stats[lifetime]['avg'])
    stats[lifetime]['std'] = np.array(stats[lifetime]['std'])

# Compute additional quantities:
# - Sum of standard deviations of average lifetime for the two lifetimes.
# - Difference in the average lifetime (4.1 ns minus 3.6 ns).
std_36 = stats[3.6]['std']
std_41 = stats[4.1]['std']
sum_std = std_36 + std_41
diff_avg = stats[4.1]['avg'] - stats[3.6]['avg']

# -----------------------------
# 4. Plotting the Results
# -----------------------------
plt.figure(figsize=(10, 6))

# Plot standard deviation of average lifetime for each fluorophore
plt.plot(time_windows, std_36, marker='o', linestyle='-', color='blue', label='Std of Avg Lifetime (3.6 ns)')
plt.plot(time_windows, std_41, marker='o', linestyle='-', color='orange', label='Std of Avg Lifetime (4.1 ns)')

# Plot sum of the standard deviations (dashed green line)
plt.plot(time_windows, sum_std, marker='o', linestyle='--', color='green', label='Sum of Std')

# Plot difference in the average lifetimes (4.1 ns - 3.6 ns) (dashed red line)
plt.plot(time_windows, diff_avg, marker='o', linestyle='--', color='red', label='Difference in Avg Lifetime')

plt.xlabel("Time Window (ns)")
plt.ylabel("Value (ns)")
plt.title("Per-Event Average Lifetime: Std, Sum of Std, and Difference vs. Time Window")
plt.legend()
plt.tight_layout()
plt.show()
