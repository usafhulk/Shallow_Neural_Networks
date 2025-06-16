from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

# Define the neural network architecture (same as the trained model)
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
model.load_state_dict(torch.load("customer_purchase_model.pth"))
model.eval()  # Set the model to evaluation mode

# Create FastAPI app
app = FastAPI()

# Define request body using Pydantic
class CustomerData(BaseModel):
    browsing_time: float

# Define prediction endpoint
@app.post("/predict/")
async def predict(customer: CustomerData):
    # Prepare input data
    input_data = torch.tensor([[customer.browsing_time]], dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_data).item()  # Get the predicted probability
    
    # Determine purchase decision
    purchase_decision = "WILL PURCHASE" if prediction > 0.5 else "WON'T PURCHASE"
    
    return {
        "browsing_time": customer.browsing_time,
        "purchase_probability": prediction,
        "decision": purchase_decision
    }

# Run the app using: uvicorn script_name:app --reload