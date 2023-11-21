"""Neural network models."""

import typing

import torch
import torch.nn as nn

ModelName: typing.TypeAlias = typing.Literal["linear-relu-stack"]


class TypedModel(nn.Module):
    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


class Model:
    def __init__(self, name: ModelName) -> None:
        match name:
            case "linear-relu-stack":
                self._model = LinearReluStack()
        self.name = name

    def parameters(self) -> typing.Iterator[nn.Parameter]:
        return self._model.parameters()

    def train(self) -> None:
        self._model.train()

    def eval(self) -> None:
        self._model.eval()

    def state_dict(self) -> dict[str, typing.Any]:
        return self._model.state_dict()

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._model(input_tensor)


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
