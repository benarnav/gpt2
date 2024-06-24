from typing import Literal

import torch
import torch.nn as nn


class GeLU(nn.Module):

    def __init__(self, approximate: Literal["none", "tanh"] = "none"):
        self.approximate = approximate
        super().__init__()

    def forward(self, x: torch.Tensor):
        if self.approximate == "tanh":
            return 0.5 * x * (1 + torch.tanh(((2 / torch.pi) ** 0.5) * (x + 0.044715 * (x**3))))

        return x * 0.5 * (1 + torch.erf(x / (2**0.5)))
