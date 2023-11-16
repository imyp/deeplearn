"""Neural network models."""

from typing import Any

import torch
import torch.nn as nn

Model = nn.Module


class TypedModel(Model):
    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


class LinearReluStack(TypedModel):
    """Simple linear relu stack with two hidden layers."""

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.flatten = nn.Flatten()
        self.relu_stack = nn.Sequential(
            nn.Linear(40 * 40 * 40, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(x)
        logits = self.relu_stack(flat)
        return logits
