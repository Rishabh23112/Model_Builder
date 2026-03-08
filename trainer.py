import torch
import torch.nn as nn
import torch.optim as optim
from dataset_analyzer import analyze_dataset
import os

def train_model(model, train_loader, task, epochs=5):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if task == "classification":
        y_train=train_loader.dataset.tensors[1].to(device)
        class_counts = torch.bincount(y_train)

        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()

        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.MSELoss()

    optimizer=optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        
        total_loss=0

        for x_batch, y_batch in train_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Reset gradients to 0
            optimizer.zero_grad()

            # Forward propagation
            outputs=model(x_batch)

            # Calculate loss
            loss=criterion(outputs, y_batch)

            # Back propagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss+=loss.item()

        avg_loss=total_loss/len(train_loader)

        print(f"Epoch {epoch+1}, Loss:{avg_loss:.4f}")

    os.makedirs("Trained_Models", exist_ok=True)
    torch.save(model.state_dict(), "Trained_Models/trained_model.pth")
    print("Model saved as Trained_Models/trained_model.pth")