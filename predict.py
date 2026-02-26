import sys
import torch
import torch.nn as nn
import json

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

def predict(sqft, age, rooms):
    try:
        model = HousePriceModel()
        model.load_state_dict(torch.load("house_model.pth", weights_only=True))
        model.eval()
        
        # Normalize inputs the exact same way as in train.py
        x = torch.tensor([[sqft / 1000.0, age / 10.0, rooms / 1.0]], dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(x)
            
        # Denormalize output
        price = pred.item() * 100000.0
        # return the predicted price formatted as JSON
        print(json.dumps({"predictedPrice": max(0, price)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Expected 3 arguments: sqft age rooms"}))
        sys.exit(1)
        
    try:
        sqft = float(sys.argv[1])
        age = float(sys.argv[2])
        rooms = float(sys.argv[3])
        predict(sqft, age, rooms)
    except ValueError:
        print(json.dumps({"error": "Arguments must be numbers"}))
        sys.exit(1)
