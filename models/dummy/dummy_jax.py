# %%
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn

# %%
# TODO: Specify the dynamic arguments for jit

@jit
def loss_mse(res, gt):
    return jnp.mean((res-gt)**2)


def get_parameters(cfg):
    key = random.PRNGKey(cfg.model.key)
    parameters = []
    sizes = cfg.model.parameter_sizes
    for size in sizes: 
        key, subkey = random.split(key)
        parameters.append(random.normal(subkey, (size), dtype=jnp.float32))
    return parameters

@jit
def model_call(data, _time, parameters, _key):
    x = data

    for parameter in parameters:
        # TODO: Reconsider ORDER in terms of shape representation
        x = jnp.matmul(parameter, x.T).T
        x = nn.sigmoid(x)
    
    return x 

@jit
def loss_fn(parameters, data, gt):
    res = model_call(data, parameters)
    loss = loss_mse(res, gt)
    return loss

def get_optim_parameters(cfg):
    # TODO: CONVERT TO OPTIMIMZER CONFIG
    return [jnp.array([cfg.optimizer.step_size], dtype=jnp.float32), grad(loss_fn, 0)]

def optim_alg(optim_parameters, parameters, data, gt):
#    for parameter, gradient in zip(parameters, gradients):
#        parameter.at[:].add(-step_size * gradient)
    grad_fn = optim_parameters[-1]
    gradients = grad_fn(parameters, data, gt)
    new_parameters = []
    for i in range(len(parameters)):
        new_parameters.append(parameters[i] - optim_parameters[0] * gradients[i])
    
    new_optim_parameters = optim_parameters
    return new_optim_parameters, new_parameters
    
def get_shapes(array):
    return [x.shape for x in array]

if __name__ == "__main__":
    # Andreas needs this to see utils.utils etc. (also in linux)
    # import sys
    # sys.path.append("/media/sf_Bsc-Diffusion")
    # end

    from utils.utils import get_hydra_config
    cfg = get_hydra_config()
    print(cfg)
    parameters = get_parameters(cfg)
    data = jnp.ones((10, 1), dtype=jnp.float32)
    imgs = model_call(data, parameters)
    key = random.PRNGKey(69)
    gt = random.normal(key, imgs.shape, dtype=jnp.float32)
    grads = grad(loss_fn,0)
    loss_grad = grads(parameters, data, gt)
    print(type(loss_grad))
    assert len(loss_grad) == len(parameters)
    for g, p in zip(loss_grad, parameters):
        print(g.shape, p.shape)
    
    c_parameters = [jnp.ones_like(p) for p in parameters]

    print(sum([(p-c).sum() for p,c in zip(parameters, c_parameters)]))
    parameters = optim_alg(cfg, parameters, loss_grad,gt)
    print(get_shapes(parameters))
    print(get_shapes(c_parameters))

#    s = 0
#    for i in range(len(parameters)):
#        s += parameters[i]-c_parameters[i]
#    print(s)
    print(sum([(p-c).sum() for p,c in zip(parameters, c_parameters)]))
    from visualization.visualize import display_images
    display_images(cfg,imgs.T)
# %%
