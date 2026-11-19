"""Native linear head with one module call per denoising objective."""

import torch
import torch.nn.functional as F
from torch import nn


class CloverDenoisingHead(nn.Linear):
    """Keep chunking inside the FSDP boundary, preserving native weight names.

    Accelerate can group the final norm and head in one FSDP2 unit. Calling
    the head once per CE chunk would revisit that unit a data-dependent number
    of times on each rank and can diverge its backward collective ordering.
    Ordinary inference still returns logits through the native linear path.
    """

    def forward(self, hidden, targets=None, *, chunk_size=128):
        if targets is None:
            return super().forward(hidden)
        if chunk_size <= 0:
            raise ValueError("CE chunk size must be positive")
        if targets.numel() == 0:
            # Even an empty objective must enter the same FSDP module boundary.
            return hidden.sum() * 0 + self.weight.reshape(-1)[0] * 0
        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), chunk_size):
            values = hidden[start:start + chunk_size]
            target = targets[start:start + chunk_size]
            loss = loss + F.cross_entropy(F.linear(values, self.weight, self.bias).float(), target, reduction="sum")
        return loss / targets.numel()
