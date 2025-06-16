# Customer Purchase Prediction API

A FastAPI-based web service for predicting customer purchase behavior using a shallow neural network built with PyTorch.

## Overview

This project provides a REST API that predicts whether a customer will make a purchase based on their browsing time. The model is a simple feedforward neural network trained to output purchase probabilities.

- **Frameworks/Libraries:** PyTorch, FastAPI, Pydantic
- **Purpose:** Predict customer purchase likelihood from browsing time
- **Deployment:** Can be run locally or in a containerized environment

## Features

- **Prediction Endpoint:** Accepts a list of browsing times and returns purchase predictions.
- **Simple Neural Network:** A shallow network with 1 hidden layer.
- **API Documentation:** Interactive Swagger UI at `/docs`.

## Project Structure

customer-purchase-prediction-api/ ├── Dockerfile ├── README.md ├── docker-compose.yml ├── requirements.txt ├── customer_purchase_prediction.py ├── models/ │ └── trained_model.pth └── src/ ├── config/ ├── models/ ├── services/ └── utils/

Code

## How It Works

1. **Input:** Browsing time(s) for one or more customers.
2. **Processing:** The API feeds the input into a trained neural network model.
3. **Output:** For each input, returns whether the customer is likely to purchase (1.0) or not (0.0), along with probabilities.

## Example Usage

### Request

`POST /predict`

```json
{
  "browsing_time": [5.0, 15.0, 30.0, 60.0]
}
Response
JSON
{
  "predictions": [0.0, 1.0, 1.0, 1.0]
}
Getting Started
Prerequisites
Python 3.7+
pip
Installation
Clone the repository:

bash
git clone https://github.com/usafhulk/Shallow_Neural_Networks.git
cd Shallow_Neural_Networks/customer-purchase-prediction-api
Install dependencies:

bash
pip install -r requirements.txt
Ensure model file exists:

The trained PyTorch model should be saved as customer_purchase_model.pth in the project directory.
Run the API:

bash
python customer_purchase_prediction.py
Access the Swagger UI:

Open http://127.0.0.1:8000/docs in your browser.
Docker Support
A Dockerfile and docker-compose.yml are included for containerized deployment.

## Experimentation Notebooks

This repository contains a series of Jupyter notebooks that document the development path from basic neural network experimentation to a deployable customer purchase prediction API. Each notebook explores core machine learning ideas and practical insights, building up the model and implementation step by step.

### Notebook Progression

- **v_1 Simple One Hidden Layer Neural Network.ipynb**
  - Introduces a basic feedforward neural network with a single hidden layer.
  - Focus: Predicts purchase likelihood from browsing time, with data visualization and accuracy evaluation.

- **v_2 Neural Networks More Hidden Neurons.ipynb**
  - Explores the power of adding more hidden neurons to the network.
  - Shows how increased capacity allows modeling more complex customer purchase patterns.

- **v_3 Neural Networks with One Hidden Layer Noisy XOR.ipynb**
  - Tackles nonlinearly separable problems (e.g., XOR-style patterns in customer behavior).
  - Demonstrates why multiple neurons and nonlinear activations are critical.

- **v_4 Multiclass Neural Networks: Customer Purchase Behavior Classification.ipynb**
  - Extends the approach to multiclass classification: No Purchase, Window Shopping, Research Purchase, Impulse Purchase.
  - Includes model comparison, validation, and decision boundary visualization.

- **v_5 Backpropogation and Vanishing Gradient Analysis.ipynb**
  - Investigates backpropagation in deep networks and highlights the vanishing gradient problem.
  - Compares sigmoid vs. ReLU activations and shows their effect on training and gradient flow.

- **v_6 Activation Functions in Neural Networks: Comprehensive Comparison.ipynb**
  - Compares sigmoid, tanh, and ReLU activation functions for both accuracy and convergence.
  - Provides recommendations for activation function selection in practice.

---

> These notebooks form the experimental foundation for the production API in `customer-purchase-prediction-api/`. They illustrate the iterative process of model selection, architecture tuning, and practical neural network considerations for real-world customer prediction tasks.

Explore each notebook in the repository root for code, plots, and detailed commentary.

Notes
The neural network is simple (1 hidden layer, 5 hidden units) and expects browsing time as the only feature.
Model architecture and training code can be adapted as needed.
Make sure to retrain and save the model if you change the architecture or input features.
Author: usafhulk

Code
Let me know if you want a more detailed explanation about any section or specific usage instructions!
