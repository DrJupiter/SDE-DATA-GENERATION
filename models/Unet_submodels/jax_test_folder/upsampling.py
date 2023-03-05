#%%
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax import vmap,grad,jit

import torch.nn.functional as F
import torch as t

import numpy as np

import matplotlib.pyplot as plt

import equinox as eqx
# %%

img = t.ones(2,3,4,5)

up = t.nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True)

up(img).shape

#%%

def upsample2d(x, factor=2):
    # stolen from https://github.com/yang-song/score_sde/blob/main/models/up_or_down_sampling.py
    _N, C, H, W = x.shape
    x = jnp.reshape(x, [-1, C, 1, H, 1, W])
    x = jnp.tile(x, [1, 1, factor, 1, factor, 1])
    return jnp.reshape(x, [-1, C, H * factor, W * factor])


img = jnp.ones((2,3,4,5))

upsample2d(img,factor=2).shape

# %%
