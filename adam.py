import math
from typing import Iterable

import torch
import torch.nn as nn


class Adam(nn.Module):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 0.0,
        lr_max: float = 2.5e-4,
        lr_warmup: int = 2000,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        total_t: int = 98000,
    ) -> None:
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.lr_max = lr_max
        self.warmup = lr_warmup
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 1
        self.total_t = total_t

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    def _get_learning_rate(self):
        if self.t <= self.warmup:
            return (self.t / self.warmup) * self.lr_max
        else:
            return self.lr_max * 0.5 * (1 + math.cos(math.pi * (self.t - self.warmup) / (self.total_t - self.warmup)))

    @torch.inference_mode
    def step(self):

        self.lr = self._get_learning_rate()
        for idx, param in enumerate(self.params):
            g_t = param.grad
            if self.weight_decay != 0:
                g_t += self.weight_decay * param

            self.m[idx] = self.betas[0] * self.m[idx] + (1 - self.betas[0]) * g_t
            self.v[idx] = self.betas[1] * self.v[idx] + (1 - self.betas[1]) * (g_t**2)
            m_hat = self.m[idx] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[idx] / (1 - self.betas[1] ** self.t)

            param -= (self.lr * m_hat) / (torch.sqrt(v_hat) + self.eps)

        self.t += 1
