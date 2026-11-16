import torch

# Input, weight and bias
x = torch.tensor(3.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# target Value
target = torch.tensor(10.0)

# Forward propogation
prediction = x * w + b

# Calculate the loss
loss = (prediction - target) ** 2

print("Prediction:", prediction.item())
print("Loss:", loss.item())

# Backpropagation
loss.backward()

# Understanding the PyTorch training sequence
# optimizer.zero_grad() # Clear previous gradients
# prediction = model(input_data) # Forward pass
# loss = loss_function(prediction, target) # Calculate the loss
# loss.backward() # Backward pass to compute gradients
# optimizer.step() # Update the weights based on gradients
