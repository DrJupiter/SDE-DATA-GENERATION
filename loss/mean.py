
from utils.utility import batch_matmul
from jax import numpy as jnp
from jax import vmap
from sde.sde import get_sde

def get_mean_matching(cfg):

    sde = get_sde(cfg)
 
    def mean_matching(func, function_parameters, data, perturbed_data, time, _z, text_embedding, key):

        #mean, cov = sde.parameters(time, data)
        mean = sde.mean(time, data)
        inv_cov = sde.diffusion.inv_covariance(time)

        difference = func(perturbed_data, time, text_embedding, function_parameters, key)-mean
        #return 0.5 * jnp.mean(batch_matmul(difference, difference))
        if sde.diffusion.diagonal_form:
            #difference_length = jnp.square(jnp.linalg.norm(vmap(lambda a,b: a * b)(-inv_cov,(difference))))
            difference_length = jnp.square(jnp.linalg.norm(-inv_cov*difference))
        else:
            difference_length = jnp.square(jnp.linalg.norm(vmap(lambda a,b: a @ b)(-inv_cov,(difference))))
        return 0.5 * jnp.mean(difference_length)
    

    return mean_matching