import torch
import torch.nn as nn
import torch.optim as optim


class HistogramClassifier(nn.Module):
    def __init__(self, num_bins=120, num_classes=3):
        super(HistogramClassifier, self).__init__()
        # Example architecture: 2 hidden layers + output
        self.network = nn.Sequential(
            nn.Linear(num_bins, 64),  # from histogram bins to 64 hidden units
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Instantiate model
model = HistogramClassifier(num_bins=120, num_classes=3)