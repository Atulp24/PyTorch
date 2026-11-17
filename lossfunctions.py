# MSE Loss
import torch
import torch.nn as nn

actual = torch.tensor([100.0, 200.0, 100.0])
predicted = torch.tensor([90.0, 180.0, 310.0])

# Calculate MSE Loss
loss_function = nn.MSELoss()
loss = loss_function(predicted, actual)

print("MSE Loss:", loss.item())

# MAE Loss
import torch
import torch.nn as nn

actual = torch.tensor([100.0, 200.0, 100.0])
predicted = torch.tensor([90.0, 180.0, 310.0])

# Calculate MAE Loss
loss_function = nn.L1Loss()
loss = loss_function(predicted, actual)

print("MAE Loss:", loss.item())

# Binary Cross-Entropy
import torch
import torch.nn as nn

actual = torch.tensor([1.0, 0.0, 1.0])
predicted = torch.tensor([0.9, 0.1, 0.8])

loss_function = nn.BCELoss()
loss = loss_function(predicted, actual)

print("Binary Cross-Entropy Loss:", loss.item())

# Note: BCELoss expects probabilities. In practical neural networks, BCEWithLogitsLoss is generally preferred because it combines sigmoid and binary cross-entropy more numerically stably.

# Cross-Entropy for classification
import torch
import torch.nn as nn

# Raw model outputs (logits)
predictions = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 0.3]
])

# Correct class indices
targets = torch.tensor([0, 1])

loss_function = nn.CrossEntropyLoss()

loss = loss_function(predictions, targets)

print("Cross-Entropy Loss:", loss.item())