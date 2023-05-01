
from jax import vmap, jacrev 
import jax
import jax.debug as jdebug
import jax.numpy as jnp
from jax import random as jrandom
from utils.utils import batch_matmul

# sharding
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

def get_implicit_score_matching(cfg):

    if cfg.loss.sharding and cfg.model.sharding:
        print("Sharding loss")
        sharding = PositionalSharding(mesh_utils.create_device_mesh((len(jax.devices()),1)))
        def implicit_score_matching(func, function_parameters, _data , perturbed_data, time, key):
            """
            func: The function, f, is assumed to be the score of a function, f = ∇_x log(p(x;θ)).
    
            x = data
            θ = function_parameters
    
            The loss that is computed is a numerical estimate of 
                E_(p_D(x))[1/2 ||f(x,θ)||^2 + Div_x(f(x,θ))]
            """
            keys = jrandom.split(key, num=int(perturbed_data.shape[0]))
            hess = jacrev(func, 0)
    
            div = lambda x, t, k: jnp.sum(jnp.diag((hess(x, t, function_parameters, k))))
            #divergence = vmap(div, (0, 0, 0), 0)(perturbed_data, time.reshape(-1,1), keys) # TODO: is vmap good here?, ask Paul?
            divergence = []
            for x, t, k in zip(perturbed_data, time.reshape(-1,1), keys):
                divergence.append(jnp.sum(jnp.diag(hess(x,t, function_parameters, k))))
    
            divergence = jnp.array(divergence) 
            # TODO do new keys
            #divergence = div(data[0], time.reshape(-1,1)[0], key)
            #divergence = jnp.array([div(x, t, key) for (x,t) in zip(data, time.reshape(-1,1))])
            #divergence = div(data[0], time.reshape(-1,1)[0])
            #print(f"The divergence {divergence}")
            #print(divergence)
            score = func(perturbed_data, time, function_parameters, key)
            score_length = jnp.square(jnp.linalg.norm(score))
            #print((jnp.linalg.norm(out)**2).shape)    
            #score = jax.device_put(func(perturbed_data, time, function_parameters, key), sharding.replicate(0))
    
        #    dot = []
        #    for i, s in enumerate(score):
        #        dot.append(jnp.dot(s,s))
        #    dot = jnp.array(dot)
            #print(batch_matmul(score, score) - jnp.einsum("bc,bc->b",score,score))
    
            #print(batch_matmul(score, score)-dot) 
            # jnp.einsum("bc,bc->b",score,score)
            return jnp.mean(0.5 * score_length + divergence)
    else:

        def implicit_score_matching(func, function_parameters, _data , perturbed_data, time, key):
            """
            func: The function, f, is assumed to be the score of a function, f = ∇_x log(p(x;θ)).

            x = data
            θ = function_parameters

            The loss that is computed is a numerical estimate of 
                E_(p_D(x))[1/2 ||f(x,θ)||^2 + Div_x(f(x,θ))]
            """
            keys = jrandom.split(key, num=int(perturbed_data.shape[0]))
            hess = jacrev(func, 0)

            div = lambda x, t, k: jnp.sum(jnp.diag((hess(x, t, function_parameters, k))))
            divergence = vmap(div, (0, 0, 0), 0)(perturbed_data, time.reshape(-1,1), keys) # TODO: is vmap good here?, ask Paul?

            # TODO do new keys
            #divergence = div(data[0], time.reshape(-1,1)[0], key)
            #divergence = jnp.array([div(x, t, key) for (x,t) in zip(data, time.reshape(-1,1))])
            #divergence = div(data[0], time.reshape(-1,1)[0])
            #print(f"The divergence {divergence}")
            #print(divergence)

            score = func(perturbed_data, time, function_parameters, key)

        #    dot = []
        #    for i, s in enumerate(score):
        #        dot.append(jnp.dot(s,s))
        #    dot = jnp.array(dot)
            #print(batch_matmul(score, score) - jnp.einsum("bc,bc->b",score,score))

            #print(batch_matmul(score, score)-dot) 
            # jnp.einsum("bc,bc->b",score,score)
            return jnp.mean(0.5 * batch_matmul(score, jnp.array(score, copy=True)) + divergence)
    return implicit_score_matching

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    def func(x, timesteps, a, key):
        print(f"The timesteps: {timesteps}")
        
        return a*x**2 
    from utils.utils import get_hydra_config
    config = get_hydra_config(overrides=["loss=implicit_sm"]) 
    implicit_score_matching = get_implicit_score_matching(config)
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
    
    