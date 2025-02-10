import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the HDF5 file into a DataFrame (replace 'your_file.h5' and 'your_key')
file_path = 't/clustering/cy3_clustered.hdf5'  # Replace with your actual file path
df = pd.read_hdf(file_path, key='locs')  # Replace 'your_key' with the correct key if needed

# Step 2: Ensure required columns exist
required_columns = {'group', 'x', 'y', 'lifetime', 'photons', 'brightness', 'bg', 'bg_over_on'}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"The following required columns are missing in the HDF5 file: {missing_columns}")


# Step 3: Define a function to calculate statistics for each group
def calculate_group_stats(group):
    # Mean center of the group
    x_center, y_center = group['x'].mean(), group['y'].mean()
    lifetime_center = group['lifetime'].mean()

    # Calculate statistics
    stats = {
        'std_x': group['x'].std(),
        'std_y': group['y'].std(),
        'distance_from_center': np.sqrt((group['x'] - x_center) ** 2 + (group['y'] - y_center) ** 2).mean(),
        'lifetime_center': lifetime_center,
        'std_lifetime': group['lifetime'].std(),
        'mean_photons': group['photons'].mean(),
        'mean_brightness': group['brightness'].mean(),
        'mean_bg': group['bg'].mean(),
        'mean_bg_over_on': group['bg_over_on'].mean()
    }
    return pd.Series(stats)


# Step 4: Apply the statistics calculation grouped by 'group'
group_stats = df.groupby('group', group_keys=False).apply(calculate_group_stats)

# Step 5: Ensure only numeric columns are used for correlation
numeric_group_stats = group_stats.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric to NaN

# Step 6: Drop columns with all NaNs (if any) before correlation
numeric_group_stats = numeric_group_stats.dropna(axis=1, how='all')

# Step 7: Correlate metrics within the group statistics
correlation_matrix = numeric_group_stats.corr()

# Step 8: Correlate std_lifetime with other metrics and display
std_lifetime_corr = correlation_matrix['std_lifetime'].drop('std_lifetime')  # Exclude self-correlation
print("\nCorrelation of std_lifetime with other metrics:")
print(std_lifetime_corr)

# Step 9: Plot the correlation matrix using matplotlib
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title("Correlation Matrix", pad=20)
plt.xticks(ticks=np.arange(correlation_matrix.shape[1]), labels=correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(correlation_matrix.shape[0]), labels=correlation_matrix.index)
plt.show()

# Step 10: Plot the average lifetime values of each group
plt.figure(figsize=(8, 6))
plt.bar(group_stats.index, group_stats['lifetime_center'], color='skyblue')
plt.xlabel('Group')
plt.ylabel('Average Lifetime')
plt.title('Average Lifetime per Group')
plt.xticks(rotation=45)
plt.show()

# Step 11: Histogram of mean lifetimes
plt.figure(figsize=(8, 6))
plt.hist(group_stats['lifetime_center'], bins=15, color='lightgreen', edgecolor='black')
plt.xlabel('Mean Lifetime')
plt.ylabel('Frequency')
plt.title('Histogram of GroupsMeanLifetimes')
plt.show()

# Step 12: Histogram of standard deviations of lifetimes
plt.figure(figsize=(8, 6))
plt.hist(group_stats['std_lifetime'], bins=15, color='lightcoral', edgecolor='black')
plt.xlabel('Standard Deviation of Lifetime')
plt.ylabel('Frequency')
plt.title('Histogram of GroupsLifetimeStandardDeviations')
plt.show()
