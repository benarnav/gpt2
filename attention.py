import math

import einops
import torch
import torch.nn as nn

from config import GPT2Config


class Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.W_Q = nn.Parameter(torch.randn((config.num_heads, config.d_model, config.d_k)) * math.sqrt(0.02))
        self.W_K = nn.Parameter(torch.randn((config.num_heads, config.d_model, config.d_k)) * math.sqrt(0.02))
        self.W_V = nn.Parameter(torch.randn((config.num_heads, config.d_model, config.d_v)) * math.sqrt(0.02))
        self.W_O = nn.Parameter(torch.randn((config.num_heads * config.d_v, config.d_model)) * math.sqrt(0.02))

        zeroes = torch.zeros((self.config.num_heads, self.config.d_seq, self.config.d_seq))
        upper_tri = torch.triu(torch.ones((self.config.num_heads, self.config.d_seq, self.config.d_seq)), diagonal=1)
        self.mask = torch.where(upper_tri == 1, torch.tensor(float("-inf")), zeroes)

    def forward(self, residual: torch.Tensor):  # shape ["batch", "seq", "d_model"]
        K = einops.einsum(self.W_K, residual, "num_heads d_model d_k, batch seq d_model -> num_heads batch seq d_k")
        Q = einops.einsum(self.W_Q, residual, "num_heads d_model d_k, batch seq d_model -> num_heads batch seq d_k")
        V = einops.einsum(self.W_V, residual, "num_heads d_model d_v, batch seq d_model -> num_heads batch seq d_v")
        QK = einops.einsum(Q, K, "num_heads batch seq_q d_k, num_heads batch seq_k d_k -> num_heads batch seq_q seq_k")
        QK_masked = QK + self.mask

        soft = torch.softmax((QK_masked / math.sqrt(self.config.d_k)), dim=-1)
        attn = einops.einsum(soft, V, "num_heads seq_q seq_k, num_heads batch seq_k d_v -> num_heads batch seq_q d_v")
        multi_head = einops.rearrange(attn, "num_heads batch seq_q d_v -> batch seq_q (num_heads d_v)")

        return einops.einsum(
            self.W_O, multi_head, "num_heads_d_v d_model, batch seq_q num_heads_d_v -> batch seq_q d_model"
        )
