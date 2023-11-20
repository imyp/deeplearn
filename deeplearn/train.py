"""Training functionality."""
import torch

import deeplearn.data as data
import deeplearn.loss as loss
import deeplearn.model as model
import deeplearn.optim as optim


def loop(
    loader: data.Loader[data.TorchTuple],
    net: model.Model,
    loss_fn: loss.LossFunction,
    optimizer: optim.Optimizer,
) -> float:
    size = len(loader.dataset)  # pyright: ignore[reportGeneralTypeIssues]
    last_batch = len(loader) - 1
    net.train()
    for batch, (X, y) in enumerate(data.iter_loader(loader)):
        pred = net(X)
        batch_loss = loss_fn(pred, y)

        batch_loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step_zero()

        if batch % 20 == 0:
            loss_float, current = batch_loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_float:>7f} [{current:>5d}/{size}]")
        if batch == last_batch:
            return batch_loss.item()
    raise RuntimeError


def test_loop(
    loader: data.Loader[data.TorchTuple], net: model.Model, loss_fn: loss.LossFunction
) -> float:
    net.eval()
    batches = len(loader)
    test_loss = 0

    with torch.no_grad():
        for X, y in data.iter_loader(loader):
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")
    return test_loss


def train_test_loops(
    epochs: int,
    train_loader: data.Loader[data.TorchTuple],
    test_loader: data.Loader[data.TorchTuple],
    optimizer: optim.Optimizer,
    net: model.Model,
    loss_fn: loss.LossFunction,
    model_name: str,
    loss_name: str,
):
    info: list[list[float]] = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = loop(train_loader, net, loss_fn, optimizer)
        test_loss = test_loop(test_loader, net, loss_fn)
        info.append([t, train_loss, test_loss])
    torch.save(  # pyright: ignore[reportUnknownMemberType]
        torch.Tensor(info), loss_name
    )
    torch.save(net.state_dict(), model_name)  # pyright: ignore[reportUnknownMemberType]
    print("Done!")
