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
input_size = 1
hidden_size = 5
output_size = 1
model = CustomerPurchaseNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("customer_purchase_model.pth"))
model.eval()  # Set the model to evaluation mode

# Create FastAPI app
app = FastAPI()

# Define request body
class CustomerRequest(BaseModel):
    browsing_time: List[float]

# Define prediction endpoint
@app.post("/predict")
def predict(customer_request: CustomerRequest):
    # Convert input to tensor
    browsing_times = torch.tensor(customer_request.browsing_time).float().unsqueeze(1)
    
    # Make predictions
    with torch.no_grad():
        purchase_probabilities = model(browsing_times).numpy().flatten()
    
    # Prepare response
    predictions = [
        {
            "browsing_time": time,
            "purchase_probability": prob,
            "prediction": "WILL PURCHASE" if prob > 0.5 else "WON'T PURCHASE"
        }
        for time, prob in zip(customer_request.browsing_time, purchase_probabilities)
    ]
    
    return {"predictions": predictions}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)