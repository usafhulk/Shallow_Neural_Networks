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
    browsing_time: List[float]  # List of browsing times

# Define prediction endpoint
@app.post("/predict")
def predict(request: CustomerRequest):
    # Convert input to tensor
    browsing_times = torch.tensor(request.browsing_time).float().unsqueeze(1)  # Shape: (N, 1)
    
    with torch.no_grad():  # Disable gradient calculation
        purchase_probabilities = model(browsing_times)  # Get predictions

    # Convert probabilities to binary predictions
    predictions = (purchase_probabilities > 0.5).float().numpy().flatten()  # Shape: (N,)
    
    # Prepare response
    response = {
        "browsing_time": request.browsing_time,
        "predictions": predictions.tolist()  # Convert to list for JSON serialization
    }
    return response

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)