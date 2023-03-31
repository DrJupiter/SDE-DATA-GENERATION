## Needed for Andras to import in a subfolder from a subfolder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
##

from models.dummy import dummy_jax
# import optax

#from models.ddpm.ddpm_unet_funcs import ddpm_unet, get_parameters as func_ddpm_unet, func_get_parameters
from models.ddpm.ddpm_unet_functional import get_ddpm_unet, get_parameters 
from models.ddpm.ddpm_unet import ddpm_unet as class_ddpm_unet


def get_model(cfg, key):
    if cfg.model.name == "dummy_jax":
        return dummy_jax.get_parameters(cfg), dummy_jax.model_call
    elif cfg.model.name == "ddpm_unet":
        if cfg.model.type == "function":
        # ddpm_model = ddpm_unet(cfg)
            params, key = get_parameters(cfg, key)
            return params, get_ddpm_unet(cfg) 
        elif cfg.model.type == "class":
            model = class_ddpm_unet(cfg)
            return model.get_parameters(cfg, key), model.forward
        else:
            raise ValueError(f"Model type {cfg.model.type} not found for {cfg.model.name}")

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