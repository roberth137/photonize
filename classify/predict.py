import torch
import numpy as np
import pandas as pd
from models import HistogramCNN  # or whichever model class you used

test_data = pd.read_hdf('cy3_1402_4p5_histogram.hdf5', key='hist')

test_np = test_data.to_numpy()

print(test_np.shape)

histograms = test_np[:, :-1]
events = test_np[:, -1]
# Create the model instance
model_infer = HistogramCNN(num_bins=120, num_classes=3)

# Load state dict
model_infer.load_state_dict(torch.load("histogram_model_test.pt", weights_only=True))
model_infer.eval()  # Set to evaluation mode

# Suppose you have new data, 'X_new' (NumPy array)
X = torch.tensor(histograms, dtype=torch.float32)

with torch.no_grad():
    outputs = model_infer(X)
    _, preds = torch.max(outputs, dim=1)
    predicted_class = preds.tolist()

for event, cls in zip(events, predicted_class):
    print(event, cls)
#print("Predicted classes:", preds)