
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn


from utils.utils import get_model_sharding
from sde.sde import get_sde
from sde.subvp import SUBVPSDE

# %%
# TODO: Specify the dynamic arguments for jit

@jit
def loss_mse(res, gt):
    return jnp.mean((res-gt)**2)


def get_parameters(cfg):
    key = random.PRNGKey(cfg.model.key)
    parameters = []
    sizes = cfg.model.parameter_sizes
    datashape =int(abs(jnp.prod(jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0]))))
    sizes[0][1] = datashape 
    sizes[-1][0] = datashape

    sizes.append([cfg.model.text_embedding, cfg.text_embedding.shape])
    initializer = nn.initializers.glorot_normal(in_axis=-1, out_axis=-2)
    for size in sizes: 
        key, subkey = random.split(key)
        parameter = initializer(subkey, (size), dtype=jnp.float32)
        parameters.append(parameter)
     
    if cfg.model.sharding:
        parameters = get_model_sharding(cfg)(parameters)
    return parameters

def get_dummy_train(cfg):

    @jit
    def model_call(data, _time, text_embedding, parameters, _key):
        x = data
        in_shape = x.shape
        embedding_dim = x.shape[0] if len(x.shape) == 1 else x.shape[1]
        embedding_dim = int(embedding_dim)
        time_emb = get_timestep_embedding(_time*999, embedding_dim)
        x = time_emb + x + jnp.matmul(parameters[-1], text_embedding.T).T

        W = parameters[:-1]
        for parameter in W[:-1]:
            # TODO: Reconsider ORDER in terms of shape representation
            x = jnp.matmul(parameter, x.T).T
            x = nn.sigmoid(x)
        x = jnp.matmul(W[-1], x.T).T
        x = x.reshape(in_shape) 
        return x

    return model_call



def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    For math behind this see paper.\n
    timesteps: array of ints describing the timestep each "picture" of the batch is perturbed to.\n
    timesteps.shape = B\n
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    \n
    Credit to DDPM (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90)
    \n I just converted it to jax.
    """
    assert len(timesteps.shape) == 1 # and timesteps.dtype == tf.int32

    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.int32) * -emb)
    emb = jnp.int32(timesteps)[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad if uneven number
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

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
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    from utils.utils import get_hydra_config
    cfg = get_hydra_config(overrides=['model=dummy', 'wandb.log.loss=false', 'wandb.log.img=false', 'visualization.visualize_img=true'])
    print(cfg)
    from models.model import get_model 
    key = random.PRNGKey(cfg.model.key)
    key, subkey = random.split(key)
    parameters, model_call = get_model(cfg, subkey)
    from data.dataload import dataload
    train, test = dataload(cfg)
    iter_train = iter(train)
    data, label = next(iter_train)
    timesteps = random.uniform(subkey, (data.shape[0],), minval=0, maxval=1)
    imgs = model_call(data, timesteps, parameters, key)
    

#    s = 0
#    for i in range(len(parameters)):
#        s += parameters[i]-c_parameters[i]
#    print(s)

    from visualization.visualize import display_images
    display_images(cfg, imgs, label)
# %%
