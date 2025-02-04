import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.data import DataLoader
from models import HistogramClassifier, HistogramCNN



##Load data
cy3_df = pd.read_hdf('training_data/cy3_histogram.hdf5', key='hist')
a550_df = pd.read_hdf('training_data/a550_histogram.hdf5', key='hist')
a565_df = pd.read_hdf('training_data/a565_histogram.hdf5', key='hist')

cy3_np = cy3_df.to_numpy()
a550_df = a550_df.to_numpy()
a565_np = a565_df.to_numpy()


array_combined = np.vstack((cy3_np, a565_np, a550_df))#, a565_np))

#df_combined = pd.concat([cy3_np, a565_np], axis=0)  # Stack rows
#array_combined = df_combined.to_numpy()

X = array_combined[:, :-1]
y = array_combined[:, -1]

print(f'X: {X}, shape(X): {X.shape}')
print(f'Y: {y}, shape(Y): {y.shape}')

print(f'created X and y arrays.')

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% validation
    random_state=42,
    stratify=y      # optional but recommended for classification
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# Create Datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

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


def validate(model, loader):
    """Return average loss & accuracy on a given DataLoader."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)  # or your chosen loss
            total_loss += loss.item()

            # Predictions
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

batch_size = 32  # or whatever you prefer



model = HistogramCNN(num_bins=X.shape[1], num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100
for epoch in range(num_epochs):  # number of epochs
    model.train()  # training mode
    running_loss = 0.0

    for histograms, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(histograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    val_loss, val_accuracy = validate(model, val_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

