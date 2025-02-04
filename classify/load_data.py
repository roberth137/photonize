import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data import DataLoader


##Load data
cy3_df = pd.read_hdf('training_data/cy3_histogram.hdf5', key='hist')
a550_df = pd.read_hdf('training_data/a550_histogram.hdf5', key='hist')
a565_df = pd.read_hdf('training_data/a565_histogram.hdf5', key='hist')

df_combined = pd.concat([cy3_df, a550_df, a565_df], axis=0)  # Stack rows
array_combined = df_combined.to_numpy()

X = array_combined[:, :-1]
y = array_combined[:, -1]
print(f'X: {X}, shape(X): {X.shape}')
print(f'Y: {y}, shape(Y): {y.shape}')


# Split the dataset into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
#X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#y_test_tensor = torch.tensor(y_test, dtype=torch.long)


class HistogramDataset(Dataset):
    def __init__(self, X, y):
        # Convert NumPy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)  # number of samples

    def __getitem__(self, idx):
        # Return one sample + label pair
        return self.X[idx], self.y[idx]

dataset = HistogramDataset(X, y)
batch_size = 32  # or whatever you prefer
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
