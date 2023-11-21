"""Loss functions that are easily configurable."""

import typing

import torch
import torch.nn as nn

LossName: typing.TypeAlias = typing.Literal["mse"]


class LossFunction:
    def __init__(self, name: LossName) -> None:
        match name:
            case "mse":
                self._loss = nn.MSELoss()
        self.name = name

    def __call__(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(predicted, target)
