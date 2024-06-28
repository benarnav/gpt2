from typing import Literal

import torch
import torch.nn as nn


class GeLU(nn.Module):
    """
    Implements the Gaussian Error Linear Unit (GeLU) activation function.

    Args:
        approximate (Literal["none", "tanh"], optional): Specifies whether to use the exact
        GeLU function or an approximation using tanh. Default is "none".

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies the GeLU activation function to the input tensor.
    """

    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        self.approximate = approximate
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the GeLU activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying GeLU activation.
        """
        if self.approximate == "tanh":
            return (
                0.5
                * x
                * (1 + torch.tanh(((2 / torch.pi) ** 0.5) * (x + 0.044715 * (x**3))))
            )

        return x * 0.5 * (1 + torch.erf(x / (2**0.5)))
