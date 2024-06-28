import math

import einops
import torch
import torch.nn as nn

from config import GPT2Config


class Attention(nn.Module):
    """
    Implements the multi-head self-attention mechanism used in GPT-2.

    Args:
        config (GPT2Config): Configuration object containing hyperparameters for the model.

    Attributes:
        W_Q (torch.nn.Parameter): Query weight matrix.
        W_K (torch.nn.Parameter): Key weight matrix.
        W_V (torch.nn.Parameter): Value weight matrix.
        W_O (torch.nn.Parameter): Output weight matrix.
        b_Q (torch.nn.Parameter): Query bias vector.
        b_K (torch.nn.Parameter): Key bias vector.
        b_V (torch.nn.Parameter): Value bias vector.
        b_O (torch.nn.Parameter): Output bias vector.
        mask (torch.Tensor): Masking tensor to prevent attention to future positions.

    Methods:
        forward(residual: torch.Tensor) -> torch.Tensor:
            Computes the multi-head self-attention output.

    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.W_Q = nn.Parameter(torch.randn((config.num_heads, config.d_model, config.d_head)) * math.sqrt(0.02))
        self.W_K = nn.Parameter(torch.randn((config.num_heads, config.d_model, config.d_head)) * math.sqrt(0.02))
        self.W_V = nn.Parameter(torch.randn((config.num_heads, config.d_model, config.d_head)) * math.sqrt(0.02))
        self.W_O = nn.Parameter(torch.randn((config.num_heads * config.d_head, config.d_model)) * math.sqrt(0.02))
        self.W_O.data /= config.num_layers**0.5  # scaling initialization as specified in the paper

        self.b_Q = nn.Parameter(torch.randn((config.num_heads, config.d_head)) * math.sqrt(0.02))
        self.b_K = nn.Parameter(torch.randn((config.num_heads, config.d_head)) * math.sqrt(0.02))
        self.b_V = nn.Parameter(torch.randn((config.num_heads, config.d_head)) * math.sqrt(0.02))
        self.b_O = nn.Parameter(torch.randn((config.d_model)) * math.sqrt(0.02))

        zeroes = torch.zeros((self.config.num_heads, self.config.d_seq, self.config.d_seq))
        upper_tri = torch.triu(
            torch.ones((self.config.num_heads, self.config.d_seq, self.config.d_seq)),
            diagonal=1,
        )
        self.mask = torch.where(upper_tri == 1, torch.tensor(float("-inf")), zeroes)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-head self-attention output.

        Args:
            residual (torch.Tensor): Input tensor with shape (batch, seq, d_model).

        Returns:
            torch.Tensor: Output tensor with shape (batch, seq, d_model).
        """
        self.mask = self.mask.to(self.W_K.device)

        K = einops.einsum(
            self.W_K,
            residual,
            "num_heads d_model d_head, batch seq d_model -> batch seq num_heads d_head",
        )
        K += self.b_K

        Q = einops.einsum(
            self.W_Q,
            residual,
            "num_heads d_model d_head, batch seq d_model -> batch seq num_heads d_head",
        )
        Q += self.b_Q

        V = einops.einsum(
            self.W_V,
            residual,
            "num_heads d_model d_head, batch seq d_model -> batch seq num_heads d_head",
        )
        V += self.b_V

        QK = einops.einsum(
            Q,
            K,
            "batch seq_q num_heads d_head, batch seq_k num_heads d_head -> batch num_heads seq_q seq_k",
        )
        QK_masked = QK + self.mask

        soft = torch.softmax((QK_masked / math.sqrt(self.config.d_head)), dim=-1)
        attn = einops.einsum(
            soft,
            V,
            "batch num_heads seq_q seq_k, batch seq_v num_heads d_head -> batch seq_q num_heads d_head",
        )
        multi_head = einops.rearrange(attn, "batch seq_q num_heads d_head -> batch seq_q (num_heads d_head)")

        return (
            einops.einsum(
                self.W_O,
                multi_head,
                "num_heads_d_head d_model, batch seq_q num_heads_d_head -> batch seq_q d_model",
            )
            + self.b_O
        )
