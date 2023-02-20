#%%
# Inspirtation from https://jax.readthedocs.io/en/latest/notebooks/convolutions.html
import jax
import jax.numpy as jnp
import jax.lax as lax

import torch.nn.functional as F
import torch as t

import numpy as np

import matplotlib.pyplot as plt

# %%
# config
img_h = 18
img_w = 18
img_channels = 3

out_channel = 3
kernel_size = 3

batch_size = 2

# input
img_j = jnp.zeros((batch_size, img_h, img_w, img_channels), dtype=jnp.float32)
for k in range(1):
  x = 1
  y = 1
  img_j = img_j.at[0, x:x+10, y:y+10, k].set(1.0)

img_np = np.zeros((batch_size, img_h, img_w, img_channels), dtype=np.float32)
for k in range(1):
  x = 1
  y = 1
  img_np[0, x:x+10, y:y+10, k] = 1.0

print("input shapes match? ",img_j.shape == img_np.shape)

# weight
# kernel shape := (out_channels,in_channels/groups ,kH,kW)
kernel_np = np.zeros((out_channel, img_channels, kernel_size, kernel_size), dtype=np.float32)
kernel_np += np.array([[1, 1, 0],
                       [1, 0,-1],
                       [0,-1,-1]])[np.newaxis, np.newaxis,:, :]

kernel_j = jnp.zeros((out_channel, img_channels, kernel_size, kernel_size), dtype=jnp.float32)
kernel_j += jnp.array([[1, 1, 0],
                       [1, 0,-1],
                       [0,-1,-1]])[jnp.newaxis, jnp.newaxis,:, :]
print("weight shapes match? ",kernel_j.shape == kernel_np.shape, kernel_j.shape)

#%%
# torch
t_conv = F.conv2d(
            t.tensor(np.transpose(img_np,[0,3,1,2])),
            t.tensor(kernel_np),
            stride=1,
            padding=1
                )
print("t_conv shape: ", t_conv.shape)
print("First output channel:")
plt.figure(figsize=(3,3))
plt.imshow(np.array(t_conv)[0,0,:,:])
#%%
# Jax input denotions
# N - batch dimension
# H - spatial height
# W - spatial height
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension

# jax
j_conv = lax.conv(
            jnp.transpose(img_j,[0,3,1,2]),    # lhs = NCHW image tensor
            kernel_j, # rhs = OIHW conv kernel tensor
            (1, 1),  # window strides
            'SAME') # padding mode
print("j_conv shape: ", j_conv.shape)
print("First output channel:")
plt.figure(figsize=(3,3))
plt.imshow(np.array(j_conv)[0,0,:,:])
#%%
print("output shapes match? ",j_conv.shape==np.array(t_conv).shape)
print("number of differences",np.sum(j_conv!=np.array(t_conv)))

# %%
print((np.array(t_conv))[0,0,:,:],"\n")
print((j_conv)[0,0,:,:])
# %%
