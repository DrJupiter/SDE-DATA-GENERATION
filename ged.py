from jax import numpy as jnp, pmap, vmap
import jax

def get_fisk(y):

    def fisk(x,w):
        return jnp.sum(jnp.matmul(x,w["lve"]["fisk"]))

    return jnp.ones(y),fisk

x,fisk = get_fisk(3)

ff = jax.jit(jax.grad(fisk,0))
w = {"lve":{"fisk": jnp.arange(12).reshape(3,4), "ged":22}}
print(ff(jnp.ones(3),w))
