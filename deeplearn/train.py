"""Training functionality."""
import torch
import torch.optim as optim

import deeplearn.data as data
import deeplearn.model as model


def loop(
    loader: data.Loader[data.TorchTuple],
    net: model.TypedModel,
    loss_fn: model.TypedModel,
    optimizer: optim.Optimizer,
):
    size = len(loader.dataset)  # pyright: ignore[reportGeneralTypeIssues]
    net.train()
    for batch, (X, y) in enumerate(data.iter_loader(loader)):
        pred = net(X)
        loss = loss_fn(pred, y)

        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_float, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_float:>7f} [{current:>5d}/{size}]")


def test_loop(
    loader: data.Loader[data.TorchTuple],
    net: model.TypedModel,
    loss_fn: model.TypedModel,
):
    net.eval()
    batches = len(loader)
    test_loss = 0

    with torch.no_grad():
        for X, y in data.iter_loader(loader):
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")
