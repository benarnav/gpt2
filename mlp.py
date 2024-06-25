import torch
import torch.nn as nn

from config import GPT2Config
from gelu import GeLU


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_hidden = config.d_hidden

        self.layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_hidden), GeLU(approximate="none"), nn.Linear(self.d_hidden, self.d_model)
        )
        nn.init.normal_(self.layers[0].weight, mean=0.0, std=0.2)
        nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.2)
        with torch.no_grad():
            self.layers[-1].weight /= config.num_layers**0.5  # scaling initialization as specified in the paper

    def forward(self, residual: float[torch.Tensor, "batch seq d_model"]):
        return self.layers(residual)
