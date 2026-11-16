import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 4),  # Input layer with 2 features, hidden layer with 4 neurons
            nn.ReLU(),        # Activation function
            nn.Linear(4, 1)   # Output layer with 1 neuron
        )

    def forward(self, x):
        return self.network(x)

model = SimpleNeuralNetwork()

input_data = torch.tensor([[0.5, 0.8]])

prediction = model(input_data)

print(prediction)