"""For plotting data"""

import matplotlib.axes as axes
import matplotlib.pyplot as pyplot
import plotly.graph_objects as graph_objects  # pyright: ignore[reportMissingTypeStubs]
import torch

import deeplearn.data as data


def plot_volume(volume_data: torch.Tensor) -> None:
    """Create a volumetric plot of values in grid."""
    volume_data_numpy = volume_data.numpy()
    assert volume_data_numpy.shape == data.GRID.x.shape
    d = graph_objects.Volume(
        x=data.GRID.x.flatten(),
        y=data.GRID.y.flatten(),
        z=data.GRID.z.flatten(),
        value=volume_data_numpy.flatten(),
        isomin=volume_data_numpy.min(),
        isomax=volume_data_numpy.max(),
        opacity=0.1,
        surface_count=20,
    )
    f = graph_objects.Figure(data=d)
    f.show()  # pyright: ignore[reportUnknownMemberType]


def loss_files(filenames: list[str], suffixes: list[str]):
    figure = pyplot.figure()  # pyright: ignore[reportUnknownMemberType]
    ax = figure.add_subplot()  # pyright: ignore[reportUnknownMemberType]
    for file_name, suffix in zip(filenames, suffixes):
        t = torch.load(file_name)  # pyright: ignore[reportUnknownMemberType]
        _plot_data_to_ax(ax, t, suffix)
    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    pyplot.show()  # pyright: ignore[reportUnknownMemberType]


def _plot_data_to_ax(ax: axes.Axes, tensor: torch.Tensor, suffix: str):
    epochs = tensor[:, 0].numpy()
    train_loss = tensor[:, 1].numpy()
    test_loss = tensor[:, 2].numpy()
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        epochs, train_loss, label=f"train-{suffix}"
    )
    ax.plot(
        epochs, test_loss, label=f"test-{suffix}"
    )  # pyright: ignore[reportUnknownMemberType]


def loss_file(filename: str):
    figure = pyplot.figure()  # pyright: ignore[reportUnknownMemberType]
    ax = figure.add_subplot()  # pyright: ignore[reportUnknownMemberType]
    t = torch.load(filename)  # pyright: ignore[reportUnknownMemberType]
    _plot_data_to_ax(ax, t, suffix="")
    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    pyplot.show()  # pyright: ignore[reportUnknownMemberType]
