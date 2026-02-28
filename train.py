import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import json
import sys
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)

class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Deep Neural Network with 3 hidden layers
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def generate_data(num_samples, noise_level=0.1):
    X = []
    y = []
    for _ in range(num_samples):
        sqft = random.uniform(500, 5000)
        age = random.uniform(0, 100)
        rooms = random.randint(1, 10)
        
        # Base price logic: 
        # $150 per sqft
        # -$500 per year of age
        # +$10000 per room
        base_price = 50000 + (sqft * 150) - (age * 500) + (rooms * 10000)
        
        # Add realistic noise
        noise = random.gauss(0, base_price * noise_level)
        price = max(10000, base_price + noise)
        
        # Normalize inputs
        X.append([sqft / 1000.0, age / 10.0, rooms / 1.0])
        y.append([price / 100000.0])
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(samples=8000, noise=0.1, epochs=50):
    print(f"Generating synthetic data (samples={samples}, noise={noise})...")
    X_train, y_train = generate_data(samples, noise)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = HousePriceModel()
    # MSE is appropriate for continuous regression
    criterion = nn.MSELoss()
    # Adam is more efficient than SGD as it uses adaptive learning rates for each parameter
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Print every 50 epochs as requested (or first/last)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
            
    # Save model and plot
    torch.save(model.state_dict(), "house_model.pth")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.savefig("client/public/loss_curve.png")
    print("Model and loss curve saved.")
    return losses

if __name__ == "__main__":
    # Allow command line overrides for demo mode
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    noise_lvl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    train_model(num_samples, noise_lvl, num_epochs)
