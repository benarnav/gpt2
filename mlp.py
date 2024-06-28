import torch
import torch.nn as nn

from config import GPT2Config
from gelu import GeLU


class MLP(nn.Module):
    """
    Implements the Feed-Forward Neural Network (MLP) used in GPT-2.

    Args:
        config (GPT2Config): Configuration object containing hyperparameters for the model.

    Attributes:
        d_model (int): Dimensionality of the model.
        d_hidden (int): Dimensionality of the hidden layer.
        layers (torch.nn.Sequential): Sequential container of the MLP layers, including
            two linear layers and a GeLU activation function.

    Methods:
        forward(residual: torch.Tensor) -> torch.Tensor:
            Applies the MLP to the input tensor.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_hidden = config.d_hidden

        self.layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_hidden),
            GeLU(approximate="none"),
            nn.Linear(self.d_hidden, self.d_model),
        )
        nn.init.normal_(self.layers[0].weight, mean=0.0, std=0.2)
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.2)
        with torch.no_grad():
            self.layers[-1].weight /= (
                config.num_layers**0.5
            )  # scaling initialization as specified in the paper

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP to the input tensor.

        Args:
            residual (torch.Tensor): Input tensor with shape (batch, seq, d_model).

        Returns:
            torch.Tensor: Output tensor with shape (batch, seq, d_model).
        """
        return self.layers(residual)
