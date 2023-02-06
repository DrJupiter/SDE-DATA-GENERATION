# %%
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random


# %%

@jit
def loss_mse(res, gt):
    return jnp.mean((res-gt)**2)

# %%

key = random.PRNGKey(0)
x = random.normal(key, (3072,), dtype=jnp.float64)
key, subkey = random.split(key)
y = random.normal(subkey, (3072,), dtype=jnp.float64)



# %%
derivate = jit(grad(loss_mse))
derivate(x,y)
# %%

def get_linear(cfg):
    key = random.PRNGKey(cfg.model.dummy.key)
    layers = []
    sizes = cfg.model.dummy.layers
    for i in range(len(sizes)-1): 
        key, subkey = random.split(key)
        layers.append(random.normal(key, (sizes[i], sizes[i+1]), dtype=jnp.float32 ))

def model_call(linear):
    return 