"""For plotting data"""

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


def plot_test_train_data(filename: str):
    t = torch.load(filename)  # pyright: ignore[reportUnknownMemberType]
    figure = pyplot.figure()  # pyright: ignore[reportUnknownMemberType]
    ax = figure.add_subplot()  # pyright: ignore[reportUnknownMemberType]
    epochs = t[:, 0].numpy()
    train_loss = t[:, 1].numpy()
    test_loss = t[:, 2].numpy()
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        epochs, train_loss, label="train"
    )
    ax.plot(epochs, test_loss, label="test")  # pyright: ignore[reportUnknownMemberType]
    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    pyplot.show()  # pyright: ignore[reportUnknownMemberType]
