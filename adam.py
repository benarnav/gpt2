import math
from typing import Iterable

import torch
import torch.nn as nn


class Adam(nn.Module):
    """
    Implements the Adam optimization algorithm with learning rate scheduling.

    Args:
        params (Iterable[nn.parameter.Parameter]): Iterable of parameters to optimize.
        lr (float, optional): Initial learning rate. Default is 0.0.
        lr_max (float, optional): Maximum learning rate. Default is 2.5e-4.
        lr_warmup (int, optional): Number of warmup steps for the learning rate schedule. Default is 2000.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.01.
        total_t (int, optional): Total number of training steps. Default is 98000.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 0.0,
        lr_max: float = 2.5e-4,
        lr_warmup: int = 2000,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
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
        """
        Sets the gradients of all optimized parameters to None.
        """
        for param in self.params:
            param.grad = None

    def _learning_rate_schedule(self) -> float:
        """
        Computes the learning rate according to the warmup and cosine decay schedule.

        Returns:
            float: The computed learning rate.
        """
        if self.t <= self.warmup:
            return (self.t / self.warmup) * self.lr_max
        else:
            return (
                self.lr_max
                * 0.5
                * (
                    1
                    + math.cos(
                        math.pi * (self.t - self.warmup) / (self.total_t - self.warmup)
                    )
                )
            )

    @torch.inference_mode
    def step(self) -> None:
        """
        Performs a single optimization step.
        """
        self.lr = self._learning_rate_schedule()
        for idx, param in enumerate(self.params):
            g_t = param.grad

            self.m[idx] = self.betas[0] * self.m[idx] + (1 - self.betas[0]) * g_t
            self.v[idx] = self.betas[1] * self.v[idx] + (1 - self.betas[1]) * (g_t**2)
            m_hat = self.m[idx] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[idx] / (1 - self.betas[1] ** self.t)

            param -= ((self.lr * m_hat) / (torch.sqrt(v_hat) + self.eps)) + (
                self.weight_decay * param
            )

        self.t += 1
