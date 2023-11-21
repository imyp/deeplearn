"""Optimizers that are easily configurable."""

import typing

import torch.optim as optim
import torch.optim.optimizer as ooptim

OptimizerName: typing.TypeAlias = typing.Literal["adam"]


class Optimizer:
    def __init__(
        self,
        algorithm: OptimizerName,
        parameters: ooptim.params_t,
        learning_rate: float,
    ) -> None:
        match algorithm:
            case "adam":
                self._optimizer = optim.Adam(params=parameters, lr=learning_rate)

    def step_zero(self) -> None:
        """Run optimizer step and zeroing of gradient"""
        self._optimizer.step()
        self._optimizer.zero_grad()
