import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.data import DataLoader
from models import HistogramClassifier, HistogramCNN, HistogramClassifierWithAttention
import time



##Load data
cy3_df = pd.read_hdf('training_data/cy3_histogram.hdf5', key='hist')
a550_df = pd.read_hdf('training_data/a550_histogram.hdf5', key='hist')
a565_df = pd.read_hdf('training_data/a565_histogram.hdf5', key='hist')

cy3_np = cy3_df.to_numpy()
a550_df = a550_df.to_numpy()
a565_np = a565_df.to_numpy()

print('here')

array_combined = np.vstack((cy3_np, a565_np, a550_df))

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

# Convert to tensors and create PyTorch datasets
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

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
    model.eval() #to evaluation mode
    correct = 0             #just
    total = 0               #some
    total_loss = 0.0        #metrics

    with torch.no_grad():               #no gradient computation since only evaluating!
        for inputs, labels in loader:                   #grabs inputs and labels from loader
            outputs = model(inputs)                     #makes predictions
            loss = F.cross_entropy(outputs, labels)     #compares predictions with ground truth
                                                        #cross_entropy most common for classification
            total_loss += loss.item()                   #adds up the loss as python integer

            # Predictions
            _, preds = torch.max(outputs, 1)            #return highest probability as output class
            correct += (preds == labels).sum().item()   #element wise comparison of preds and labels, summed up and converted to integer
            total += labels.size(0)                     #returns the size of teh training sample

    avg_loss = total_loss / len(loader)                 # calculates average loss
    accuracy = correct / total                          # calculates accuracy
    return avg_loss, accuracy                           # returns loss and accuracy

# Set hyperparameters and create model
#batch_size = 32                                         # use powers of 2 since GPUs are optimized for 2^n batch sized
model = HistogramCNN(num_bins=X.shape[1], num_classes=3)    # choose model class to be trained and define parameters
criterion = nn.CrossEntropyLoss()                       # define CrossEntropy loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)     # define optimizer
num_epochs = 10                                       # number of training epochs


for epoch in range(num_epochs):                         # iterate over training epochs
    model.train()                                       # sets model to training mode
    running_loss = 0.0                                  # tracking cumulative loss over all batches in one epoch

    for histograms, labels in train_loader:             # iterate over batches of data provided by train_loader
        optimizer.zero_grad()                           # sets all gradients to 0, since loss.backward() adds to existing gradients
        outputs = model(histograms)                     # inference: Making the actual prediction and assigning it to output
        loss = criterion(outputs, labels)               # Calculate the loss for the predictions
        loss.backward()                                 # Backpropagation: calculate gradient of loss wrt to trainable parameters
        optimizer.step()                                # Adapt network weights -> going one step towards the loss minimum in multidimensional hyperspace
        running_loss += loss.item()                     # Add loss from current batch to running loss over epoch
    train_loss = running_loss / len(train_loader)       # Calculates average loss for this epoch

    val_loss, val_accuracy = validate(model, val_loader)    # Validates the model on unseen data

    print(f"Epoch [{epoch + 1}/{num_epochs}] "      # Print progress: Helpful to see what's actually happening 
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

torch.save(model.state_dict(), "histogram_model_test.pt")    # saves the model weights and biases in file.
                                                                # NO saving of model architecture -> define model structure before loading again!!! 

print(model.state_dict().keys())  # See what’s stored

