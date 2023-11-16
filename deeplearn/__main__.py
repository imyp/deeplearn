"""Plot dataset."""
import torch.nn as nn
import torch.optim as optim

import deeplearn.data as data
import deeplearn.model as model
import deeplearn.train as train
import deeplearn.plot as plot

learning_rate = 1e-3
batch_size = 10
epochs = 20
train_loader = data.Loader(data.FileSphereDataset("train_data"), batch_size)
test_loader = data.Loader(data.FileSphereDataset("test_data"), batch_size)
net = model.LinearReluStack()
loss_function: model.TypedModel = (
    nn.MSELoss()
)  # pyright: ignore[reportGeneralTypeIssues]
# optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
model_name="relu-linear-stack.pt"
loss_name = "epoch_train_test_adam.data"
train.train_test_loops(
    epochs,
    train_loader,
    test_loader,
    optimizer,
    net,
    loss_function,
    model_name,
    loss_name
)
plot.plot_test_train_data(loss_name)