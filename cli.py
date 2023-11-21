"""Command line interface"""

import hydra
import hydra.utils as hutils
import omegaconf
import deeplearn.optim as optim
import deeplearn.model as model
import deeplearn.loss as loss
import deeplearn.data as data
import deeplearn.train as train_module
import deeplearn.plot as plot
import pathlib
import sys 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: omegaconf.DictConfig)->None:
    """Train model from configurations specified in configuration file."""
    network: model.Model = hutils.instantiate(cfg.model)
    partial_optim = hutils.instantiate(cfg.partial_optim)
    optimizer: optim.Optimizer = partial_optim(parameters=network.parameters())
    loss_fn: loss.LossFunction = hutils.instantiate(cfg.loss)
    partial_data = hutils.instantiate(cfg.partial_data)
    datasets: data.Data = partial_data(base=pathlib.Path(hutils.get_original_cwd()))
    train_module.train(cfg.epochs, datasets, optimizer, network, loss_fn)

def generate_data():
    """Generate a train and test dataset."""
    data_dir = pathlib.Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    data.SphereDataset.generate(1000).to_file(str(data_dir/"train"))
    data.SphereDataset.generate(500).to_file(str(data_dir/"test"))

def plot_dir():
    if len(sys.argv) != 2:
        raise RuntimeError("Expected one argument.")
    file_path = pathlib.Path(sys.argv[1])
    if not file_path.exists():
        raise RuntimeError(f"File {file_path} does not exist.")
    plot.loss_file(str(file_path))
