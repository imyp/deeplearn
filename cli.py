"""Command line interface"""

import hydra
import hydra.utils as hutils
import omegaconf
import deeplearn.optim as optim
import deeplearn.model as model
import deeplearn.loss as loss
import deeplearn.data as data
import deeplearn.train as train_module
import pathlib

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