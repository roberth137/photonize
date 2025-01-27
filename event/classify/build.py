import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Example raw histogram data (each row is a histogram sample with N bins)
X = np.array([
    [12, 45, 23, 67, 89, 23, 13, 7, 0, 1],
    [9, 10, 15, 22, 30, 40, 20, 10, 5, 2],
    # Add more samples...
])

# Corresponding class labels (0, 1, or 2)
y = np.array([0, 1, 2, ...])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)