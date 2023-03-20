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
#%%
# config
img_h = 18
img_w = 18
img_channels = 3

out_channel = 3
kernel_size = 3

batch_size = 2

# input # Batchsize, Channels, H, W
img = jnp.zeros((batch_size, img_channels, img_h, img_w), dtype=jnp.float32)
img = img.at[0, 0, 2:2+10, 2:2+10].set(1.0) 

# run maxpool on batch
maxpool2d = eqx.nn.MaxPool2d(2,2)
y = vmap(maxpool2d,axis_name="batch")(img) 
y.shape

#%%
# Compare to pytorch, to see if it does the same
img_t = np.zeros((batch_size, img_channels, img_h, img_w), dtype=np.float32)
img_t[0, 0, 2:2+10, 2:2+10] = 1.0
print("identical shapes:",img.shape==img_t.shape)
print("Number of different values in original img:",np.sum(img!=img_t))


t_mp = t.nn.MaxPool2d(2,stride=2)
y_t = t_mp(t.tensor(img_t))
y_t.shape

print("identical outcome?:", np.sum(y!=np.array(y_t))==0)

# %%
