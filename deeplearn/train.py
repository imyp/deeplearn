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
) -> float:
    size = len(loader.dataset)  # pyright: ignore[reportGeneralTypeIssues]
    last_batch = len(loader) - 1
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
        if batch == last_batch:
            return loss.item()
    raise RuntimeError


def test_loop(
    loader: data.Loader[data.TorchTuple],
    net: model.TypedModel,
    loss_fn: model.TypedModel,
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
    net: model.TypedModel,
    loss_fn: model.TypedModel,
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
