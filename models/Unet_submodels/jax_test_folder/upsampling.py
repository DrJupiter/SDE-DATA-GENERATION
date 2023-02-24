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
