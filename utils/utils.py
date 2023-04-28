
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf

def get_hydra_config(config_path='../configs', job_name='test', version_base='1.3', config='defaults.yaml', overrides=[], reload=True):
    """
    !!! This function is meant for TESTING PURPOSES, not to be used
    in production. !!!

    Load the hydra config manually.
    
    (The parameters are the same as loading a hydraconfig normally)
    """

    if reload:
       hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path, job_name=job_name, version_base=version_base) 
    cfg = compose(config, overrides=overrides)
    return cfg

import jax.numpy as jnp
from jax import vmap, jit, pmap
from jax.nn import sigmoid

batch_matmul = vmap(lambda a,b: jnp.matmul(a.T, b), (0, 0) , 0)
"""
    dim(a): B x W x L1 | B x L1 
    dim(b): B x W x L2 | B x L1 

    Performs matmull across the batch dimension B.

    return: B x L1 x L2 | B  
"""

#### Rescaling x to logits or img ####

def logit(p):
    """logit function using the natrual log"""
    return jnp.log(p/(1-p))

def rescale_to_logit(cfg,x):
    """Transforms x to logit space\n
    CITE: 4.3 in https://arxiv.org/pdf/1705.07057.pdf """
    lamb = float(cfg.dataset.lamb)
    return logit(lamb + (1-2*lamb)*x)

def rescale_logit_to_img(cfg,z):
    """Transforms logit into [0,256]\n
    CITE: E2 in https://arxiv.org/pdf/1705.07057.pdf"""
    lamb = float(cfg.dataset.lamb)
    return logit(lamb + (1-2*lamb)*(z/256))

def rescale_sigmoid_img(x):
    factor = 0.05 # -500;500 -> 0;10
    max_val = 255
    return max_val/(1+jnp.exp(-1,factor))
import pickle
import os

def load_paramters(cfg, model_paramters, optimizer_paramters):

    if cfg.parameter_loading.model:
        if os.path.isdir(cfg.parameter_loading.model_path):
            with open(os.path.join(cfg.parameter_loading.model_path, f"{cfg.model.name}-parameters.pickle"), "rb") as mp:
                iteration, model_paramters = pickle.load(mp)
        
        elif os.path.isfile(cfg.parameter_loading.model_path):
            with open(cfg.parameter_loading.model_path, "rb") as mp:
                iteration, model_paramters = pickle.load(mp)
        else:
            raise FileNotFoundError(f"Unable to find {cfg.parameter_loading.model_path}")

        print(f"Loaded model: {cfg.model.name} paramters @ checkpoint iteration {iteration}")

    if cfg.parameter_loading.optimizer:
        if os.path.isdir(cfg.parameter_loading.optimizer_path):
            with open(os.path.join(cfg.parameter_loading.optimizer_path, f"{cfg.optimizer.name}-parameters.pickle"), "rb") as op:
                iteration, optimizer_paramters = pickle.load(op)

        elif os.path.isfile(cfg.parameter_loading.optimizer_path):
            with open(cfg.parameter_loading.optimizer_path, "rb") as op:
                iteration, optimizer_paramters = pickle.load(op)
        else:
            raise FileNotFoundError(f"Unable to find {cfg.parameter_loading.optimizer_path}")

        print(f"Loaded optimizer: {cfg.optimizer.name} paramters @ checkpoint iteration {iteration}")

    return model_paramters, optimizer_paramters