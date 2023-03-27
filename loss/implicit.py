import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from jax import jacfwd, vmap, jacrev
import jax.numpy as jnp
from jax import random as jrandom
from utils.utils import batch_matmul


def implicit_score_matching(func, function_parameters, data, time, key):
    """
    func: The function, f, is assumed to be the score of a function, f = ∇_x log(p(x;θ)).

    x = data
    θ = function_parameters

    The loss that is computed is a numerical estimate of 
        E_(p_D(x))[1/2 ||f(x,θ)||^2 + Div_x(f(x,θ))]
    """
    hess = jacrev(func, 0)
    div = lambda x, t, key: jnp.sum(jnp.diag(hess(x, t, function_parameters, key)))
    #divergence = vmap(div, (0, 0), 0)(data, time.reshape(-1,1)) # TODO: is vmap good here?, ask Paul?

    # TODO do new keys
    divergence = div(data[0], time.reshape(-1,1)[0], key)
    #divergence = jnp.array([div(x, t, key) for (x,t) in zip(data, time.reshape(-1,1))])
    #divergence = div(data[0], time.reshape(-1,1)[0])
    print(f"The divergence {divergence}")
    #print(divergence)

    score = func(data, time, function_parameters, key)

    return jnp.mean(0.5 * batch_matmul(score, score) + divergence)

if __name__ == "__main__":
    def func(x, timesteps, a, key):
        print(f"The timesteps: {timesteps}")
        
        return a*x**2 
    from utils.utils import get_hydra_config
    config = get_hydra_config() 
    from data.dataload import dataload
    train, _ = dataload(config)
    iter_train = iter(train)
    data, label = next(iter_train) 
    key = jrandom.PRNGKey(0)
    key, subkey = jrandom.split(key)
    data = jnp.array([jnp.linspace((1), (10), 10) + _ for _ in range(config.train_and_test.train.batch_size)])
    timesteps = jrandom.uniform(subkey, (data.shape[0],), minval=0, maxval=1)
    print(data)
    a = 1
    print(data)
    print(f"Function value of data: {func(data, timesteps, a, key)}")
    print(f"Implicit Score matching loss: {implicit_score_matching(func, a, data, timesteps, subkey)}")
    
    