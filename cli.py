"""Command line interface"""

import hydra
import hydra.utils as hutils
import omegaconf
import deeplearn.optim as optim
import deeplearn.model as model
import deeplearn.loss as loss

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: omegaconf.DictConfig)->None:
    """Train model from configurations specified in configuration file."""
    print(omegaconf.OmegaConf.to_yaml(cfg))
    network: model.Model = hutils.instantiate(cfg.model)
    optimizer: optim.Optimizer = hutils.instantiate(cfg.optimizer)
    optimizer.initialize(network.parameters())
    loss_fn: loss.LossFunction = hutils.instantiate(cfg.loss)

    print(type(optimizer))
    print(type(network))
    print(type(loss_fn))
    # TODO: Create configuration for a data class
