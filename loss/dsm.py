
from utils.utils import batch_matmul
from jax import numpy as jnp
from sde.sde import get_sde

def get_denosing_score_matching(cfg):

    sde = get_sde(cfg)
    parametric_score = sde.score


    def denosing_score_matching(func, function_parameters, data, perturbed_data, time, key):
        # Todo lambda(t)
        difference = func(perturbed_data, time, function_parameters, key)-parametric_score(data, time, perturbed_data)
        return 0.5 * jnp.mean(batch_matmul(difference, difference))
    

    return denosing_score_matching