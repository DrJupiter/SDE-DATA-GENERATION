
from utils.utility import batch_matmul
from jax import numpy as jnp
from sde.sde import get_sde

def get_mean_matching(cfg):

    sde = get_sde(cfg)
 
    def mean_matching(func, function_parameters, data, perturbed_data, time, _z, text_embedding, key):

        mean, _cov = sde.parameters(time, data)

        difference = func(perturbed_data, time, text_embedding, function_parameters, key)-mean
        #return 0.5 * jnp.mean(batch_matmul(difference, difference))
        difference_length = jnp.square(jnp.linalg.norm(difference))
        return 0.5 * jnp.mean(difference_length)
    

    return mean_matching