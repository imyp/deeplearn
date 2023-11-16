"""Plot dataset."""
import torch.nn as nn
import torch.optim as optim

import deeplearn.data as data
import deeplearn.model as model
import deeplearn.train as train

learning_rate = 1e-3
batch_size = 2
epochs = 5
size = 1000
dataset = data.SphereDataset(size)
loader = data.Loader(dataset, batch_size)
net = model.LinearReluStack()
loss_function: model.TypedModel = (
    nn.MSELoss()
)  # pyright: ignore[reportGeneralTypeIssues]
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train.loop(loader, net, loss_function, optimizer)
    train.test_loop(loader, net, loss_function)
print("Done!")