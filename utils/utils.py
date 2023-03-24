
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
from jax import vmap

batch_matmul = vmap(lambda a,b: jnp.matmul(a.T, b), (0, 0) , 0)
"""
    dim(a): B x W x L1 | B x L1 
    dim(b): B x W x L2 | B x L1 

    Performs matmull across the batch dimension B.

    return: B x L1 x L2 | B  
"""