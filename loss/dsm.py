
from utils.utils import batch_matmul
from jax import numpy as jnp
from sde.sde import get_sde

def get_denosing_score_matching(cfg):

    sde = get_sde(cfg)
    parametric_score = sde.score
    if cfg.loss.lambda_scale:

        def denosing_score_matching(func, function_parameters, data, perturbed_data, time, key):
            # Todo lambda(t)
            score_model = func(perturbed_data, time, function_parameters, key)
            parametric_estimate = parametric_score(data, time, perturbed_data)
            lambda_t = 1/jnp.mean(batch_matmul(parametric_estimate, parametric_estimate))
            difference = score_model-parametric_estimate
            return 0.5 * jnp.mean(batch_matmul(difference, difference)) * lambda_t
    
    else:
        def denosing_score_matching(func, function_parameters, data, perturbed_data, time, key):
            difference = func(perturbed_data, time, function_parameters, key)-parametric_score(data, time, perturbed_data)
            return 0.5 * jnp.mean(batch_matmul(difference, difference))
    

    return denosing_score_matching