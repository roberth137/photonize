import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import helper

filename = '3green_rfp_event.hdf5'
events = pd.read_hdf(filename, key='locs')

#pd.set_option('display.max_columns', None)  # Show all columns
#print(events)
#print(events.describe(include='all'))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming your DataFrame is named df and has the columns 'lifetime_10ps' and 'brightness_norm'
data = events[['lifetime_10ps', 'brightness_norm']]

# Optionally: Standardize the data if the scales differ significantly
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
events_scale = scaler.fit_transform(data)
# Use data_scaled in the k-means fitting instead of data

# Perform k-means clustering with 2 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(events_scale)

# Add the cluster labels to the original DataFrame
events['cluster'] = clusters
helper.dataframe_to_picasso(events, filename=filename, extension='_k_means')

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(events['lifetime_10ps'], events['brightness_norm'], c=events['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('Lifetime (10ps)')
plt.ylabel('Brightness (Normalized)')
plt.title('K-means Clustering of Events')
plt.colorbar(label='Cluster')
plt.show()