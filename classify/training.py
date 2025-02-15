import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#Load training data (labeled)
cy3_df = pd.read_hdf('training_data/cy3_histogram.hdf5', key='hist')
a550_df = pd.read_hdf('training_data/a550_histogram.hdf5', key='hist')
a565_df = pd.read_hdf('training_data/a565_histogram.hdf5', key='hist')

#Combine them to one np array
cy3_np = cy3_df.to_numpy()
a550_np = a550_df.to_numpy()
a565_np = a565_df.to_numpy()
array_combined = np.vstack((cy3_np, a565_np, a550_np))


X = array_combined[:, :-1]
y = array_combined[:, -1]

print(f'X: {X}, shape(X): {X.shape}')
print(f'Y: {y}, shape(Y): {y.shape}')

print(f'created X and y arrays.')

#split data in train and test
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% validation
    random_state=42,
    stratify=y      # optional but recommended for classification
)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

# Load PyTorch datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

def train_model(model, train_loader, val_loader=None, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()  # for 3-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()  # training mode
        running_loss = 0.0

        for histograms, labels in train_loader:
            # histograms shape: (batch_size, 150)
            # labels shape: (batch_size,) with class indices [0..2]

            optimizer.zero_grad()
            outputs = model(histograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss = {avg_loss:.4f}")

        # Optionally evaluate on validation set
        if val_loader is not None:
            evaluate(model, val_loader)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for histograms, labels in loader:
            outputs = model(histograms)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100.0 * correct / total:.2f}%")