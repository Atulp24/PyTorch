import torch

# initial parameters
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Input and target
x = torch.tensor(3.0)
target = torch.tensor(10.0)

# Learning rate
learning_rate = 0.01

# Forward propagation
prediction = x * w + b

# Calculate the loss
loss = (prediction - target) ** 2

# Backpropagation
loss.backward()

# Update parameters using gradient descent
with torch.no_grad(): # Disable gradient tracking for parameter updates
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

print("Updated weight (w):", w.item())
print("Updated bias (b):", b.item())

# Using PyTorch's SGD optimizer
# pyTorch provide an optimizer called SGD (Stochastic Gradient Descent)

import torch
import torch.nn as nn

# Create a simple model
model = nn.Linear(1, 1) # A linear layer with 1 input feature and 1 output feature

# Define optimizer
optimizer = torch.optim.SGD( # Stochastic Gradient Descent optimizer
    model.parameters(), # Parameters of the model to be optimized
    lr=0.01 # Learning rate for the optimizer
)

# Input and target
x = torch.tensor([[3.0]])
target = torch.tensor([[10.0]])

# Forward pass
prediction = model(x)

# Calculate loss
loss = nn.MSELoss()(prediction, target)

# Backpropagation
optimizer.zero_grad() # Clear previous gradients
loss.backward()

# Update parameters
optimizer.step() # Update the weights based on gradients

print("Loss:", loss.item())