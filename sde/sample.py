import jax.random as jrandom
import jax.numpy as jnp
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree, ItoMilstein

def sample(t, t0, t1, dt0, drift, diffusion, args, y0, key, tol=1e-3, reverse=True):
    brownian_motion = VirtualBrownianTree(t0, t1, tol=tol, shape=y0.shape, key=key)
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    #terms = (ODETerm(drift), ControlTerm(diffusion, brownian_motion))
    #solver = ItoMilstein()
    solver = Euler() 
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
    if config.dataset.name == 'cifar10':
        label = [config.dataset.classes[int(idx)] for idx in label]
    t = jrandom.uniform(subkey, (data.shape[0],), minval=0, maxval=1).reshape(-1, 1)
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
    #print(model(data, t, param, subkey))
    rev_drift = sde.reverse_drift(xt[0], t.reshape(-1,1)[0], [model, param if config.model.name != "sde" else data[0], subkey])
    rev_diffusion = sde.reverse_diffusion(xt[0], t.reshape(-1,1)[0], [model, param if config.model.name != "sde" else data[0], subkey])
    print(rev_drift.shape, rev_diffusion.shape)
    key, *subkey = jrandom.split(key, 3)
    drift = lambda t,y, args: sde.reverse_drift(y, jnp.array([t]), args)
    diffusion = lambda t,y, args: sde.reverse_diffusion(y, jnp.array([t]), args)
    #drift = lambda t, y, args: -y
    #diffusion = lambda t, y, args: jnp.identity(y.shape[0])
    from utils.utils import rescale_logit_to_img, rescale_to_logit
    #x_in  = rescale_to_logit(config, xt[0])
    x_in = xt[0]
    x0 = sample(0, 0, float(t[0]), -1/1000, drift , diffusion, [model, param if config.model.name != "sde" else data[0], subkey[0]], x_in, subkey[1] )
    from visualization.visualize import display_images
    display_images(config, [data[0], x_in, x0], titles=label*3)
    