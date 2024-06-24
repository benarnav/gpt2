import torch


def dropout(input: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    assert 0 <= p <= 1
    if p == 1:
        return torch.zeros_like(input)
    mask = (torch.rand(input.shape) > p).float()
    out = input * mask / (1.0 - p)

    return out
