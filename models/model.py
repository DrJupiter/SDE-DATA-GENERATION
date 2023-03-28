## Needed for Andras to import in a subfolder from a subfolder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
##

from models.dummy import dummy_jax
# import optax
from models.ddpm.ddpm_unet_funcs import ddpm_unet, get_parameters
# from loss.sumloss import sum_diff_loss
# from optimizer.optimizers import adam


def get_model(cfg, key):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_parameters(cfg), dummy_jax.model_call
    elif cfg.model.name == "ddpm_unet":
        # ddpm_model = ddpm_unet(cfg)
        params, key = get_parameters(cfg, key)
        return params, ddpm_unet 
    raise ValueError(f"Model {cfg.model.name} not found")

# def get_optim(cfg,params=None):
#     if cfg.model.name == "dummy_jax":
#         return dummy_jax.get_optim_parameters(cfg), dummy_jax.optim_alg
#     elif cfg.model.optimizer == "adam":
#         optimizer = optax.adam(learning_rate=cfg.model.learningrate)
#         assert params is not None, "pls give func get_optim the model parameters, for this optimiazation function"
#         opt_param = optimizer.init(params)
#         return optimizer, opt_param
#     raise ValueError(f"Model {cfg.model.name} not found")

# def get_loss(cfg):
#     if cfg.model.name == "dummy_jax":
#         return dummy_jax.loss_fn
#     elif cfg.model.name == "ddpm_unet":
#         return sum_diff_loss
#     raise ValueError(f"Model {cfg.model.name} not found")
    

# from utils.utils import get_hydra_config
# cfg = get_hydra_config()
# ddpm_model = ddpm_unet(cfg)