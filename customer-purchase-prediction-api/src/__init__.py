# app.py

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
input_size = 1  # One input feature: browsing time
hidden_size = 5  # Number of neurons in hidden layer
output_size = 1  # One output: purchase probability

# Initialize the model and load the weights
model = CustomerPurchaseNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("customer_purchase_model.pth"))  # Load your trained model
model.eval()  # Set the model to evaluation mode

# Create FastAPI app
app = FastAPI()

# Define request body model
class CustomerRequest(BaseModel):
    browsing_time: List[float]

# Define response model
class PurchasePrediction(BaseModel):
    customer_id: int
    browsing_time: float
    purchase_probability: float
    prediction: str

@app.post("/predict", response_model=List[PurchasePrediction])
async def predict_purchase(request: CustomerRequest):
    predictions = []
    
    # Process each browsing time
    for i, time in enumerate(request.browsing_time):
        input_tensor = torch.tensor([[time]], dtype=torch.float32)  # Prepare input tensor
        with torch.no_grad():  # Disable gradient calculation
            purchase_prob = model(input_tensor).item()  # Get prediction
            prediction = "WILL PURCHASE" if purchase_prob > 0.5 else "WON'T PURCHASE"  # Make binary decision
            
        # Append prediction to the results
        predictions.append(PurchasePrediction(
            customer_id=i + 1,
            browsing_time=time,
            purchase_probability=purchase_prob,
            prediction=prediction
        ))
    
    return predictions

# To run the FastAPI app, use the command:
# uvicorn app:app --reload