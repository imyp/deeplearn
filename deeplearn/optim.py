"""Optimizers that are easily configurable."""

import typing

import torch.optim as optim
import torch.optim.optimizer as ooptim

OptimizerName: typing.TypeAlias = typing.Literal["adam"]
ALLOWED_OPTIMIZER_NAMES = ("adam",)


class Optimizer:
    def __init__(self, algorithm: OptimizerName, learning_rate: float) -> None:
        self._optimizer: optim.Optimizer | None = None
        self._name: OptimizerName = algorithm
        self._learning_rate = learning_rate

    def initialize(self, parameters: ooptim.params_t) -> None:
        match self._name:
            case "adam":
                self._optimizer = optim.Adam(params=parameters, lr=self._learning_rate)

    def step_zero(self) -> None:
        """Run optimizer step and zeroing of gradient"""
        if self._optimizer is None:
            raise ValueError(
                "Cannot step or zero grad because optimizer is not initalized."
            )
        self._optimizer.step()
        self._optimizer.zero_grad()
