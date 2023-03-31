import jax.random as jrandom
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

def sample(t, t0, t1, dt0, drift, diffusion, args, y0, key, tol=1e-3, reverse=True):
    brownian_motion = VirtualBrownianTree(t0, t1, tol=tol, shape=y0.shape, key=key)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    solver = Euler() # Ito something soemthing
    saveat = SaveAt(dense=True)

    if reverse:
        sol = diffeqsolve(terms, solver, t1, t0, dt0=dt0, y0=y0, saveat=saveat, args=args)
        return sol.evaluate(t)
    
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, saveat=saveat, args=args)
    return sol.evaluate(t)

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    from sde import get_sde
    from utils.utils import get_hydra_config
    config = get_hydra_config(overrides=["visualization.visualize_img=true","wandb.log.img=false"]) 
    sde = get_sde(config)
    from data.dataload import dataload
    train, _ = dataload(config)
    iter_train = iter(train)
    key = jrandom.PRNGKey(0)
    key, subkey = jrandom.split(key)
    data, label = next(iter_train) 
    data = jnp.array(data.numpy(), dtype=jnp.float32)
    t = jrandom.uniform(subkey, (data.shape[0],), minval=0, maxval=1)
    print(t)
    key, subkey = jrandom.split(key)
    from models.model import get_model
    param, model = get_model(config, subkey)
    from jax.tree_util import Partial
    drift = Partial(sde.reverse_drift, sm=model)
    key, subkey = jrandom.split(key)
    xt = sde.sample(t, data, subkey)
    key, subkey = jrandom.split(key)
    print(xt.shape)
    print(model(data, t, param, subkey))
    print(sde.reverse_drift(xt[0], t[0], [model, param, subkey]))
    #sample(0, 0, t[0], -1/1000, drift, diffusion, param, xt[0],subkey )

    