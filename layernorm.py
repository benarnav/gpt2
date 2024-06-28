import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Implements Layer Normalization.

    Args:
        normalized_shape (int): The shape of the input tensor to be normalized.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.

    Attributes:
        gamma (torch.nn.Parameter): Learnable scale parameter.
        beta (torch.nn.Parameter): Learnable shift parameter.
        eps (float): Value added to the denominator for numerical stability.

    Methods:
        forward(residual: torch.Tensor) -> torch.Tensor:
            Applies layer normalization to the input tensor.
    """

    def __init__(self, normalized_shape: int, eps=1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Applies layer normalization to the input tensor.

        Args:
            residual (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        mean = torch.mean(residual, dim=-1, keepdim=True)
        var = torch.var(residual, unbiased=False, dim=-1, keepdim=True)
        y = (residual - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        return y
