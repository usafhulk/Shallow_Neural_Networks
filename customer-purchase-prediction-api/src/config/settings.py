# Import necessary libraries
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Define the neural network architecture
class CustomerPurchaseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomerPurchaseNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_output = torch.relu(self.hidden(x))
        purchase_probability = torch.sigmoid(self.output(hidden_output))
        return purchase_probability

# Load the trained model
input_size = 1
hidden_size = 5
output_size = 1
model = CustomerPurchaseNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("customer_purchase_model.pth"))  # Load your trained model
model.eval()  # Set the model to evaluation mode

# Create FastAPI app
app = FastAPI()

# Define request body model
class CustomerRequest(BaseModel):
    browsing_time: List[float]

# Define prediction endpoint
@app.post("/predict")
def predict(request: CustomerRequest):
    # Convert input to tensor
    browsing_times = torch.tensor(request.browsing_time).float().unsqueeze(1)
    
    # Get predictions
    with torch.no_grad():
        purchase_probabilities = model(browsing_times).numpy().flatten()
    
    # Prepare response
    predictions = [
        {
            "browsing_time": time,
            "purchase_probability": prob,
            "prediction": "WILL PURCHASE" if prob > 0.5 else "WON'T PURCHASE"
        }
        for time, prob in zip(request.browsing_time, purchase_probabilities)
    ]
    
    return {"predictions": predictions}

# Run the app with: uvicorn script_name:app --reload