import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Load the HDF5 file
file_name = 'Cy3_histogram.hdf5'  # Update with your actual file path
key_name = 'hist'  # Dataset key used during saving

# Read data from HDF5 using h5py
with h5py.File(file_name, 'r') as h5f:
    data = h5f[key_name][()]  # Load dataset as a NumPy array

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Assuming the last column is the 'label', remove it before computing averages
if df.shape[1] > 1:
    feature_columns = df.iloc[:, :-1]  # All columns except the last one
else:
    feature_columns = df  # If no label column, use the full dataset

# Calculate average values of each column
average_values = feature_columns.mean()

# Plot the average values as a bar chart
plt.figure(figsize=(12, 6))
average_values.plot(kind='bar')
plt.title('Max Values of Histogram Columns')
plt.xlabel('Column Index')
plt.ylabel('Max Value')
plt.grid(axis='y')
plt.show()