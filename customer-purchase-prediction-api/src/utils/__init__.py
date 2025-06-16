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

# Load the trained model (make sure to save your model after training)
model = CustomerPurchaseNet(input_size=1, hidden_size=5, output_size=1)
model.load_state_dict(torch.load("customer_purchase_model.pth"))  # Load your trained model
model.eval()  # Set the model to evaluation mode

# Initialize FastAPI
app = FastAPI()

# Define input data model
class CustomerInput(BaseModel):
    browsing_time: List[float]

# Define output data model
class PurchasePrediction(BaseModel):
    customer_id: int
    browsing_time: float
    purchase_probability: float
    prediction: str

@app.post("/predict", response_model=List[PurchasePrediction])
async def predict_purchase(input_data: CustomerInput):
    predictions = []
    
    with torch.no_grad():  # Disable gradient calculation
        for i, time in enumerate(input_data.browsing_time):
            # Prepare input tensor
            input_tensor = torch.tensor([[time]], dtype=torch.float32)
            # Get prediction
            purchase_prob = model(input_tensor).item()
            prediction = "WILL PURCHASE" if purchase_prob > 0.5 else "WON'T PURCHASE"
            # Append result
            predictions.append(PurchasePrediction(
                customer_id=i + 1,
                browsing_time=time,
                purchase_probability=purchase_prob,
                prediction=prediction
            ))
    
    return predictions

# Run the application with: uvicorn app:app --reload