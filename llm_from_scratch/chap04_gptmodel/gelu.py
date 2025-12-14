import torch
import torch.nn as nn

class GELU(nn.Module):
    """
    An implementation of the GELU (Gaussian Error Linear Unit) activation function.
    This is an approximation of the exact GELU formulation, which is computationally
    expensive. This approximation is widely used in models like GPT.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        The forward pass implements the GELU approximation formula:
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

