from jax import jacfwd
import jax.numpy as jnp
from jax import random as jrandom

def implicit_score_matching(func, function_parameters, data):
    """
    func: The function, f, is assumed to be the score of a function, f = ∇_x log(p(x;θ)).

    x = data
    θ = function_parameters

    The loss that is computed is a numerical estimate of 
        E_(p_D(x))[1/2 ||f(x,θ)||^2 + Div_x(f(x,θ))]
    """
    div_fn = jacfwd(func, argnums=0)

    div = div_fn(data, function_parameters)
    score = func(data, function_parameters)

    # TODO: Add axis to this
    jnp.mean(0.5 * jnp.dot(score, score.T) + div)

if __name__ == "__main__":
    func = lambda x, a: a*x**2    

    from utils.utils import get_hydra_config
    config = get_hydra_config() 
    from data.dataload import dataload
    train, _ = dataload(config)
    iter_train = iter(train)
    key = jrandom.PRNGKey(0)
    key, subkey = jrandom.split(key)
    data, label = next(iter_train) 
    implicit_score_matching(func, 1, data)
    
    