"""Plot dataset."""
import torch.nn as nn
import torch.optim as optim

import deeplearn.data as data
import deeplearn.model as model
import deeplearn.plot as plot
import deeplearn.train as train

learning_rate = 1e-3
batch_size = 10
epochs = 20
train_data = data.SphereDataset.generate(20).to_file("train")
test_data = data.SphereDataset.generate(10).to_file("test")
train_loader = data.Loader(data.SphereDataset.from_file("train"), batch_size)
test_loader = data.Loader(data.SphereDataset.from_file("test"), batch_size)
net = model.LinearReluStack()
loss_function: model.TypedModel = (
    nn.MSELoss()
)  # pyright: ignore[reportGeneralTypeIssues]
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
model_name = "relu-linear-stack.pt"
loss_name = "epoch_train_test_adam.data"
train.train_test_loops(
    epochs,
    train_loader,
    test_loader,
    optimizer,
    net,
    loss_function,
    model_name,
    loss_name,
)
volumes = next(v for v, _ in data.iter_loader(train_loader))
plot.plot_volumes(volumes)