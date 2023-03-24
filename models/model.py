## Needed for Andras to import in a subfolder from a subfolder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
##

from models.dummy import dummy_jax
import optax
from models.ddpm.ddpm_unet import ddpm_unet
from loss_functions.loss_functions import sum_diff_loss
from optimizer.optimizers import adam


def get_model(cfg):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_parameters(cfg), dummy_jax.model_call
    elif cfg.model.name == "ddpm_unet":
        ddpm_model = ddpm_unet(cfg.model)
        params, key = ddpm_model.get_parameters(cfg, key = cfg.model.key)
        return params, ddpm_model.forward 
    raise ValueError(f"Model {cfg.model.name} not found")

def get_optim(cfg,params=None):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_optim_parameters(cfg), dummy_jax.optim_alg
    elif cfg.model.optimizer == "adam":
        optimizer = optax.adam(learning_rate=cfg.model.learningrate)
        opt_param = optimizer.init(params)
        return optimizer, opt_param
    raise ValueError(f"Model {cfg.model.name} not found")

def get_loss(cfg):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.loss_fn
    elif cfg.model.name == "ddpm_unet":
        return sum_diff_loss
    raise ValueError(f"Model {cfg.model.name} not found")
    

# from utils.utils import get_hydra_config
# cfg = get_hydra_config()
# ddpm_model = ddpm_unet(cfg)