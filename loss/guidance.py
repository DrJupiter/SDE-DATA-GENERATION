
from utils import batch_matmul
from jax import numpy as jnp
from sde.sde import get_sde

def get_guidance_matching(cfg):

    sde = get_sde(cfg)
    parametric_score = sde.score
 
    def guidance_matching(func, function_parameters, data, perturbed_data, time, _z, text_embedding, key):
        difference = func(_z, jnp.ones_like(time), text_embedding, function_parameters, key)-parametric_score(data, jnp.ones_like(time), _z)
        #return 0.5 * jnp.mean(batch_matmul(difference, difference))
        difference_length = jnp.square(jnp.linalg.norm(difference))
        return 0.5 * jnp.mean(difference_length)
    

    return guidance_matching