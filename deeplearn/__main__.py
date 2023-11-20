import deeplearn.data as data
import deeplearn.loss as loss
import deeplearn.model as model
import deeplearn.optim as optim
import deeplearn.train as train


def run_training(
    learning_rate: float,
    batch_size: int,
    train_name: str,
    test_name: str,
    epochs: int,
    model_name: str,
    loss_name: str,
):
    train_loader = data.Loader(data.SphereDataset.from_file(train_name), batch_size)
    test_loader = data.Loader(data.SphereDataset.from_file(test_name), batch_size)
    net = model.Model("linear-relu-stack")
    loss_function = loss.LossFunction("mse")
    optimizer = optim.Optimizer("adam", learning_rate)
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


learning_rate = 1e-3
batch_size = 10
epochs = 20
train_name = "train"
test_name = "test"
model_name = "relu-linear-stack.pt"
loss_name = "epoch_train_test_adam.data"
