import torch
import torch.nn as nn
import torch.optim as optim

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