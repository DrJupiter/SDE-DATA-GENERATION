from jax import jacfwd, vmap
import jax.numpy as jnp
from jax import random as jrandom
from utils.utils import batch_matmul


def implicit_score_matching(func, function_parameters, data):
    """
    func: The function, f, is assumed to be the score of a function, f = ∇_x log(p(x;θ)).

    x = data
    θ = function_parameters

    The loss that is computed is a numerical estimate of 
        E_(p_D(x))[1/2 ||f(x,θ)||^2 + Div_x(f(x,θ))]
    """
    hess = jacfwd(func, 0)
    div = lambda x: jnp.sum(jnp.diag(hess(x, function_parameters)))
    divergence = vmap(div, (0), 0)(data)

    score = func(data, function_parameters)

    return jnp.mean(0.5 * batch_matmul(score, score) + divergence)

if __name__ == "__main__":
    func = lambda x, a: a*x**2 + jnp.sum(x)

    from utils.utils import get_hydra_config
    config = get_hydra_config() 
    from data.dataload import dataload
    train, _ = dataload(config)
    iter_train = iter(train)
    key = jrandom.PRNGKey(0)
    key, subkey = jrandom.split(key)
    data, label = next(iter_train) 
    data = jnp.array([jnp.linspace((1), (10), 10) + _ for _ in range(config.train_and_test.train.batch_size)])
    print(data)
    a = 1
    print(data)
    print(f"Function value of data: {func(data, a)}")
    print(f"Implicit Score matching loss: {implicit_score_matching(func, a, data)}")
    
    