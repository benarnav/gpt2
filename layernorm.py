import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, residual: torch.Tensor):
        mean = torch.mean(residual, dim=-1, keepdim=True)
        var = torch.var(residual, unbiased=False, dim=-1, keepdim=True)
        y = (residual - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        return y
