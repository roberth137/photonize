from sklearn.cluster import KMeans
import pandas as pd

filename = 't/or'
events = pd.read_hdf('t/orig')
pd.set_option('display.max_columns', None)  # Show all columns
print(df)
df.describe(include='all')

#kmeans = KMeans(n_clusters=3, random_state=42)
#clusters = kmeans.fit_predict(data)