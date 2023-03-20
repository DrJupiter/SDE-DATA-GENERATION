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
key = random.PRNGKey(0)

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

batchnorm = eqx.experimental.BatchNorm(
                        input_size=img_channels, #takes in-channels as input_size
                        axis_name="batch",
                        momentum=0.99,
                        eps=1e-05,
                        ) 
# updates its running statistics as a side effect of its forward pass
# BatchNorm layer must be used inside of a vmap or pmap with a matching axis_name
y = vmap(batchnorm,axis_name="batch")(img) 
y.shape


# %%
# Compare to pytorch, to see if it does the same
img_t = np.zeros((batch_size, img_channels, img_h, img_w), dtype=np.float32)
img_t[0, 0, 2:2+10, 2:2+10] = 1.0
print("identical shapes:",img.shape==img_t.shape)
print("Number of different values in original img:",np.sum(img!=img_t))


t_bn = t.nn.BatchNorm2d(
                        num_features=img_channels, #takes in-channels as input_size
                        momentum=0.01,
                        eps=1e-05,
                        affine=False
                        ) 
y_t = t_bn(t.tensor(img_t))
y_t.shape

print("Almost identical output:", np.allclose(y,np.array(y_t.detach())))


# %%
