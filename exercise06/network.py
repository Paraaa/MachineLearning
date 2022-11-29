import torch
import torch.nn.functional as F


class NeuralNetwork(torch.nn.Module):
    """A simple neral network to predict
    the targets of the XOR problem
    """

    def __init__(self):
        """Initialize the network by
        constructing each layer. This model
        is very simple and just uses on hidden
        layer (layer1) and and output layer (layer2)
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(2, 2)
        self.layer2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        """Forward the input through
        each layer and activation function
        to determine the output.

        Args:
            x (torch.Tensor()): features
        Returns:
            torch.Tensor(): return a prediction
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x
