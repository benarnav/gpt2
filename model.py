import torch
import torch.nn as nn

from attention import Attention
from config_test import GPT2Config
from layernorm import LayerNorm
from mlp import MLP


def _dropout(input: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    assert 0 <= p <= 1
    if p == 1:
        return torch.zeros_like(input)
    mask = (torch.rand(input.shape, device=input.device) > p).float()
    out = input * mask / (1.0 - p)

    return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: GPT2Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_norm = LayerNorm(config.d_model)
        self.attention = Attention(config)
        self.mlp_norm = LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, residual: torch.Tensor):
        pre_attn_norm = self.attention_norm(residual)
        attn_out = self.attention(pre_attn_norm)
        if self.training:
            attn_out = _dropout(attn_out, self.config.dropout_rate)

        residual += attn_out

        pre_mlp_norm = self.mlp_norm(residual)
        mlp_out = self.mlp(pre_mlp_norm)
        if self.training:
            mlp_out = _dropout(mlp_out, self.config.dropout_rate)

        residual += mlp_out

        return residual


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config = GPT2Config()):
        super().__init__()
        self.config = config
        self.W_E = nn.Embedding(self.config.d_vocab, self.config.d_model)
        nn.init.normal_(self.W_E.weight, mean=0.0, std=self.config.weight_init)
        self.pos = nn.Parameter(torch.zeros(config.d_seq, self.config.d_model))
        layers = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]
        layers.append(LayerNorm(self.config.d_model))
        self.layers = nn.Sequential(*layers)
        self.W_U = nn.Linear(self.config.d_model, self.config.d_vocab, bias=False)
        self.W_U.weight = self.W_E.weight

    def forward(self, input: torch.Tensor):  # "batch seq"
        embeded = self.W_E(input)
        residual = embeded + self.pos
        layers_out = self.layers(residual)

        return self.W_U(layers_out)


if __name__ == "__main__":
    m = GPT2()
    config = GPT2Config()
    tens = torch.randint(low=0, high=config.d_vocab, size=(config.batch_size, config.d_seq))
    m.forward(tens)
