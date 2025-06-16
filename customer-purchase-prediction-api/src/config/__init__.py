from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

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
model = CustomerPurchaseNet(input_size=1, hidden_size=5, output_size=1)
model.load_state_dict(torch.load('customer_purchase_model.pth'))
model.eval()  # Set the model to evaluation mode

# Create FastAPI app
app = FastAPI()

# Define request body
class CustomerRequest(BaseModel):
    browsing_time: float

# Define prediction endpoint
@app.post("/predict/")
def predict_purchase(request: CustomerRequest):
    # Prepare input tensor
    browsing_time_tensor = torch.tensor([[request.browsing_time]])
    
    # Get prediction
    with torch.no_grad():
        purchase_probability = model(browsing_time_tensor).item()
    
    # Determine purchase decision
    prediction = "WILL PURCHASE" if purchase_probability > 0.5 else "WON'T PURCHASE"
    
    return {
        "browsing_time": request.browsing_time,
        "purchase_probability": purchase_probability,
        "prediction": prediction
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)