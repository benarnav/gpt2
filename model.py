import torch
import torch.nn as nn

from attention import Attention
from config import GPT2Config
from layernorm import LayerNorm
from mlp import MLP


def _dropout(input: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """
    Apply dropout to the input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The dropout probability. Defaults to 0.1.

    Returns:
        torch.Tensor: The output tensor after applying dropout.

    Raises:
        AssertionError: If p is not between 0 and 1 inclusive.
    """
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

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            residual (torch.Tensor): The input tensor with shape (batch, seq, d_model).

        Returns:
            torch.Tensor: The output tensor after passing through the transformer block with shape (batch, seq, d_model).
        """
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
    def __init__(self, config: GPT2Config = GPT2Config()) -> None:
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT2 model.

        Args:
            input (torch.Tensor): The input tensor of shape (batch, seq).

        Returns:
            torch.Tensor: The output logits.
        """

        embeded = self.W_E(input)
        residual = embeded + self.pos
        layers_out = self.layers(residual)

        return self.W_U(layers_out)

    @torch.no_grad()
    def generate(
        self,
        input: torch.Tensor,
        max_tokens: int = 150,
        temperature: float = 1.0,
        p: float = 0.9,
    ) -> torch.tensor:
        """
        Generate text using the GPT2 model with nucleus sampling.

        Args:
            input (torch.Tensor): The input tensor to start generation from.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 150.
            temperature (float): The temperature for logits scaling. Defaults to 1.0.
            p (float): The cumulative probability threshold for nucleus sampling. Defaults to 0.9.

        Returns:
            torch.Tensor: The generated sequence of tokens.
        """

        for _ in range(max_tokens):
            logits = self(input)[:, -1, :]

            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            nucleus = torch.where(cumulative_probs > p)[0][0].item() + 1
            top_probs = sorted_probs[:nucleus]
            top_indices = sorted_indices[:nucleus]

            top_probs /= top_probs.sum()

            next_token_index = torch.multinomial(top_probs, num_samples=1)
            next_token = top_indices[next_token_index]

            input = torch.cat([input, next_token], dim=1)

            if next_token.item() == self.config.eos_token_id:
                break

        return input


# if __name__ == "__main__":
#     m = GPT2()
#     config = GPT2Config()
#     tens = torch.randint(
#         low=0, high=config.d_vocab, size=(config.batch_size, config.d_seq)
#     )
#     m.forward(tens)
