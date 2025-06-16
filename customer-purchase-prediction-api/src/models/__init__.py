# customer_purchase_prediction.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the dataset class
class CustomerDataset(Dataset):
    """Dataset containing customer browsing time and purchase decisions"""
    
    def __init__(self, num_samples=1000):
        self.browsing_time = torch.linspace(0.5, 120.0, num_samples).unsqueeze(1)
        purchase_logits = (self.browsing_time * 0.1 - 2.0).squeeze()
        noise = torch.randn(num_samples) * 0.5
        purchase_logits += noise
        self.purchased = (torch.sigmoid(purchase_logits) > 0.5).float().unsqueeze(1)
        self.len = num_samples
    
    def __getitem__(self, index):
        return self.browsing_time[index], self.purchased[index]
    
    def __len__(self):
        return self.len

# Define the neural network class
class CustomerPurchaseNet(nn.Module):
    """Neural network to predict customer purchase behavior"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomerPurchaseNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        hidden_output = torch.relu(self.hidden(x))
        purchase_probability = torch.sigmoid(self.output(hidden_output))
        return purchase_probability

# Create and train the model
def train_model():
    dataset = CustomerDataset(num_samples=800)
    input_size = 1
    hidden_size = 5
    output_size = 1
    model = CustomerPurchaseNet(input_size, hidden_size, output_size)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_x, batch_y in DataLoader(dataset, batch_size=32, shuffle=True):
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
    
    return model

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = train_model()

# Define request body for prediction
class PurchaseRequest(BaseModel):
    browsing_time: float

# Define prediction endpoint
@app.post("/predict/")
async def predict(request: PurchaseRequest):
    with torch.no_grad():
        input_tensor = torch.tensor([[request.browsing_time]])
        prediction = model(input_tensor).item()
        purchase_decision = "WILL PURCHASE" if prediction > 0.5 else "WON'T PURCHASE"
        return {
            "browsing_time": request.browsing_time,
            "purchase_probability": prediction,
            "decision": purchase_decision
        }

# To run the FastAPI app, use the command:
# uvicorn customer_purchase_prediction:app --reload