# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax
import jax

# Equinox
import equinox as eqx

######################## Basic building blocks ########################

def naive_downsample_2d(x, factor=2):
  _N, H, W, C = x.shape
  x = jnp.reshape(x, [-1, H // factor, factor, W // factor, factor, C])
  return jnp.mean(x, axis=[2, 4])

def upsample2d(x, factor=2):
    """
    Upsamling function\n
    Dublicates values to increase shape\n
    credit to SDE paper: https://github.com/yang-song/score_sde/blob/main/models/up_or_down_sampling.py
    """ 
    # get shapes of input
    B, H, W, C = x.shape
    # rehape it to BxHx1xWx1xC
    x = jnp.reshape(x, [-1, H, 1, W, 1, C])
    
    # dubilicate values factor number of times
    x = jnp.tile(x, [1, 1, factor, 1, factor, 1])

    # Collect the shapes back into the desized "shape", resulting in and increase in H and W, by the factors magnitude.
    return jnp.reshape(x, [-1, H * factor, W * factor, C])

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
        # assert len(timesteps.shape) == 1 # and timesteps.dtype == tf.int32

        half_dim = embedding_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.int32) * -emb)
        # emb = jnp.int32(timesteps)[:, None] * emb[None, :]
        emb = jnp.int32(timesteps) * emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=0)
        # emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:  # zero pad if uneven number
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        assert emb.shape == (embedding_dim,)
        return emb

class resnet_ff(eqx.Module):
    conv_layers: list
    linear_layers: list

    def __init__(self, cfg, in_channel, out_channel, embedding_dim, key) -> None:
        """Initialises the ResNet Block, see paper for further explanation of this class"""

        keys = jax.random.split(key, 4)

        conv0 = eqx.nn.Conv(num_spatial_dims = 2, kernel_size=[3,3], in_channels = in_channel[0], out_channels = out_channel[0], key = keys[0])
        conv1 = eqx.nn.Conv(num_spatial_dims = 2, kernel_size=[3,3], in_channels = out_channel[0], out_channels = out_channel[0], key = keys[1])

        self.conv_layers = [conv0,conv1]

        self.linear_layers = [
            eqx.nn.Linear(embedding_dim, out_channel[0]*out_channel[1], key=keys[2]),
            eqx.nn.Linear(in_channel[0]*in_channel[1], out_channel[0]*out_channel[1], key=keys[2])
            ]

    def __call__(self, x_in, embedding, parameters, subkey):
        C,H,W = x_in.shape
        # store local parameter "location" for use in forward 

        # Get linear parameters needed later

        # Apply the function to the input data
        x = x_in+0.0 # to counter memory share problems
        # vmap(batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1) # batchnorm
        x = nn.relu(x)
        x = self.conv_layers[0](x)
        x = nn.relu(x)
        x = x + self.linear_layers[0](x.reshape(-1))[None, None, :] # introducing time embedding
        x = x.reshap(-1,H,W)
        #x = vmap(batchnorm1,axis_name="batch")(x.transpose(0,3,2,1)).transpose(0,3,2,1) 
        x = nn.relu(x)
        #x = dropout(x,key = subkey)
        x = self.conv_layers[1](x)
        
        # perform skip connection
        x_in = self.linear_layers[1](x.reshape(-1))
        x = x.reshap(-1,H,W)

        # add skip connections
        return x+x_in

class attention(eqx.Module):
    attn_layer: list

    def __init__(self, cfg, in_channel, out_channel, embedding_dim, key) -> None:

        self.attn_layer = [eqx.nn.MultiheadAttention(num_heads = 1, query_size = in_channel[0], dropout_p = 0.0, inference = False, key = key)]

    def __call__(self, x_in, embedding, parameters, subkey):

        x = x_in+0.0 # batchnorm
        x = x.reshape(1,-1)
        x = self.attn_layer[0](query = x, key_ = x, value = x)

        # sum and return
        return x+x_in

class time_embed(eqx.Module):
    linear_layers: list

    def __init__(self, embedding_in_dim, embedding_dim, key):
        keys = jax.random.split(key, 2)
        self.linear_layers = [
            eqx.nn.Linear(embedding_in_dim, embedding_dim, key=keys[0]),
            eqx.nn.Linear(embedding_dim, embedding_dim, key=keys[1])
            ]

    def __call__(self, timesteps, embedding_dims):
        x = get_timestep_embedding(timesteps, embedding_dim = embedding_dims)
        x = self.linear_layers[0](x)
        x = self.linear_layers[1](x)
        return x

######################## Advanced building blocks ########################

class down_resnet(eqx.Module):
    resnet_layers: list
    maxpool_factor: int

    def __init__(self, cfg, in_channel, out_channel, embedding_dim, key, maxpool_factor) -> None:

        self.maxpool_factor = maxpool_factor

        resnet1 = resnet_ff(cfg, in_channel, out_channel, embedding_dim, key)
        resnet2 = resnet_ff(cfg, out_channel, out_channel, embedding_dim, key)

        self.resnet_layers = [resnet1, resnet2]

    def __call__(self, x_in, embedding, parameters, subkey) :

        keys = jax.random.split(subkey, 2)

        x0 = self.resnet_layers[0](x_in, embedding, parameters, keys[0])
        x1 = self.resnet_layers[1](x0, embedding, parameters, keys[1])
        x2 = naive_downsample_2d(x1, factor = self.maxpool_factor)

        return x0,x1,x2

class down_resnet_attn(eqx.Module):
    resnet_layers: list
    maxpool_factor: int
    attn_layers: list

    def __init__(self, cfg, in_channel, out_channel, embedding_dim, key, maxpool_factor) -> None:

        keys = jax.random.split(key, 4)
        self.maxpool_factor = maxpool_factor

        resnet1 = resnet_ff(cfg, in_channel, out_channel, embedding_dim, keys[0])
        resnet2 = resnet_ff(cfg, out_channel, out_channel, embedding_dim, keys[1])
        self.resnet_layers = [resnet1, resnet2]

        attn1 = attention(cfg, out_channel, out_channel, embedding_dim, keys[2])
        attn2 = attention(cfg, out_channel, out_channel, embedding_dim, keys[3])
        self.attn_layers = [attn1,attn2]

    def __call__(self, x_in, embedding, parameters, subkey) :

        keys = jax.random.split(subkey, 4)

        x0_1 = self.resnet_layers[0](x_in, embedding, parameters, keys[0])
        x0 = self.attn_layers[0](x0_1, embedding, parameters, keys[1])

        x1_1 = self.resnet_layers[1](x0, embedding, parameters, keys[2])
        x1 = self.attn_layers[1](x1_1, embedding, parameters, keys[3])
        
        x2 = naive_downsample_2d(x1, factor = self.maxpool_factor)

        return x0,x1,x2

class up_resnet(eqx.Module):
    resnet_layers: list
    maxpool_factor: int

    def __init__(self, cfg, in_channel, out_channel, embedding_dim, key, maxpool_factor) -> None:

        self.maxpool_factor = maxpool_factor

        resnet1 = resnet_ff(cfg, in_channel, out_channel, embedding_dim, key)
        resnet2 = resnet_ff(cfg, out_channel, out_channel, embedding_dim, key)
        resnet3 = resnet_ff(cfg, out_channel, out_channel, embedding_dim, key)

        self.resnet_layers = [resnet1, resnet2, resnet3]

    def __call__(self, x_in, x_res0, x_res1, x_res2, embedding, parameters, subkey) :

        x = self.resnet_layers[0](jnp.concatenate([x_in,x_res0],axis=-1), embedding, parameters, subkey)
        x = self.resnet_layers[1](jnp.concatenate([x,x_res1],axis=-1), embedding, parameters, subkey)
        x = self.resnet_layers[2](jnp.concatenate([x,x_res2],axis=-1), embedding, parameters, subkey)
        x = upsample2d(x, factor = self.maxpool_factor)

        return x

class up_resnet_attn(eqx.Module):
    resnet_layers: list
    maxpool_factor: int
    attn_layers: list

    def __init__(self, cfg, in_channel, out_channel, embedding_dim, key, maxpool_factor) -> None:

        keys = jax.random.split(key, 6)
        self.maxpool_factor = maxpool_factor

        resnet1 = resnet_ff(cfg, in_channel, out_channel, embedding_dim, keys[0])
        resnet2 = resnet_ff(cfg, out_channel, out_channel, embedding_dim, keys[1])
        resnet3 = resnet_ff(cfg, out_channel, out_channel, embedding_dim, keys[2])
        self.resnet_layers = [resnet1, resnet2, resnet3]

        attn1 = attention(cfg, out_channel, out_channel, embedding_dim, keys[3])
        attn2 = attention(cfg, out_channel, out_channel, embedding_dim, keys[4])
        attn3 = attention(cfg, out_channel, out_channel, embedding_dim, keys[5])
        self.attn_layers = [attn1,attn2,attn3]

    def __call__(self, x_in, x_res0, x_res1, x_res2, embedding, parameters, subkey) :

        keys = jax.random.split(subkey, 6)

        x = self.resnet_layers[0](jnp.concatenate([x_in,x_res0],axis=-1), embedding, parameters, keys[0])
        x = self.attn_layers[0](x, embedding, parameters, keys[1])

        x = self.resnet_layers[1](jnp.concatenate([x,x_res1],axis=-1), embedding, parameters, keys[2])
        x = self.attn_layers[1](x, embedding, parameters, keys[3])

        x = self.resnet_layers[2](jnp.concatenate([x,x_res2],axis=-1), embedding, parameters, keys[4])
        x = self.attn_layers[2](x, embedding, parameters, keys[5])
        
        x = upsample2d(x, factor = self.maxpool_factor)

        return x

