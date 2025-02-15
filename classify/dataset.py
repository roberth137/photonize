import torch
from torch.utils.data import Dataset

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