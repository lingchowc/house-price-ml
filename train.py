import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import os

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)

class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def generate_data(num_samples):
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
        
        # Add 10% realistic noise
        noise = random.gauss(0, base_price * 0.1)
        price = max(10000, base_price + noise) # ensuring price doesn't go below 10k
        
        # Normalize inputs for better convergence
        X.append([sqft / 1000.0, age / 10.0, rooms / 1.0])
        y.append([price / 100000.0])
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

if __name__ == "__main__":
    print("Generating synthetic data...")
    X_train, y_train = generate_data(8000)
    X_test, y_test = generate_data(2000)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = HousePriceModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training model...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")
        
    # Save the model
    torch.save(model.state_dict(), "house_model.pth")
    print("Model saved to house_model.pth")
