
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax

import equinox as eqx


#%%
class fisk():
    def sum_logistic(self,x,y):
        return jnp.sum(x**2*y)

fsk = fisk()
x_small = jnp.arange(3.)
y_small = jnp.arange(3.)
derivative_fn = grad(jit(fsk.sum_logistic),0)
derivative_fn2 = grad(jit(fsk.sum_logistic),1)
print(derivative_fn(x_small,y_small),derivative_fn2(x_small,y_small))




import string

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  print(einsum_str)
  return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)



# print(X.shape,We.shape)

# jnp.sum(jnp.einsum('bhwc,bHWc->bHWc', X, We))==jnp.sum(jnp.einsum('bhwc,bHWc->bhwc', X, We))
# jnp.sum(q) == jnp.sum(jnp.einsum('bhwc,bHWC->bhWC', X, We))

#%%
hs = [nn.conv2d(x, name='conv_in', num_units=128)] # ch = 128, ch_mult=[1,2,4,4]
for i_level in range(num_resolutions): # 4

    # Residual blocks for this resolution
    for i_block in range(num_res_blocks): # 2
        h = resnet_block(hs[-1],out_ch=ch * ch_mult[i_level]) # 4*2 = 8 times total on down # out = 128,256,512,512

        if h.shape[1] in attn_resolutions: # 16 
            h = attn_block(h) # 2 times on down
        hs.append(h)

    # Downsample
    if i_level != num_resolutions - 1: # not on last one
        hs.append(downsample(hs[-1], name='downsample', with_conv=resamp_with_conv)) # (4-1) = 3 times


#%%

for i_level in reversed(range(num_resolutions)): # 4 (3,2,1,0)

    # Residual blocks for this resolution
    for i_block in range(num_res_blocks + 1): # 2 + 1 = 3
        h = resnet_block(tf.concat([h, hs.pop()], axis=-1),out_ch=ch * ch_mult[i_level]) # ch = 128, ch_mult=[1,2,4,4]       i_lvl = 3 -> out_ch=512, hs.pop().shape = BxHxWx512

        if h.shape[1] in attn_resolutions:
            h = attn_block(h)

    # Upsample
    if i_level != 0: # Upsample every time but the last
        h = upsample(h)

#%%
 

a = jnp.arange(2*3*5*6).reshape(2,3,5,6)
b = jnp.arange(2*11*5*6).reshape(2,11,5,6)
c = jnp.concatenate((a,b),axis=1)
c.shape

#%%
h = nn.conv2d(x, name='conv_in', num_units=ch) # 0
hs = [h]  
for i_level in range(4):

    # Residual blocks for this resolution
    for i_block in range(2):
        h = resnet_block(
        hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level])
        if h.shape[1] in attn_resolutions:
            h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
        hs.append(h)

    # Downsample
    if i_level != num_resolutions - 1:
        hs.append(downsample(hs[-1])


### DOWN
h = nn.conv2d(x, name='conv_in', num_units=ch) 
hs = [h]  #0

## Res i_level = 0 32x32
# i_block = 0
h = resnet_block(hs[-1], temb=temb, out_ch=128)
hs.append(h) # 1
# i_block = 1
h = resnet_block(hs[-1], temb=temb, out_ch=128)
hs.append(h) # 2
hs.append(downsample(hs[-1])) # 32x32 -> 16x16 # 3
## 16x16


## Res i_level = 1 16x16
# i_block = 0
h = resnet_block(hs[-1], temb=temb, out_ch=256)
hs.append(h) # 4
# i_block = 1
h = resnet_block(hs[-1], temb=temb, out_ch=256)
hs.append(h) # 5
hs.append(downsample(hs[-1])) # 16x16 -> 8x8 # 6
## 8x8


## Res i_level = 2 8x8
# i_block = 0
h = resnet_block(hs[-1], temb=temb, out_ch=512)
hs.append(h) # 7
# i_block = 1
h = resnet_block(hs[-1], temb=temb, out_ch=512)
hs.append(h) # 8
hs.append(downsample(hs[-1])) # 8x8 -> 4x4 # 9
## 4x4


## Res i_level = 3 4x4
# i_block = 0
h = resnet_block(hs[-1], temb=temb, out_ch=512)
hs.append(h) # 10 #4x4
# i_block = 1
h = resnet_block(hs[-1], temb=temb, out_ch=512)
hs.append(h) # 11 # 4x4
# hs.append(downsample(hs[-1]))
## 4x4


### Upsampling
# pop 11
h = resnet_block(tf.concat([h, hs.pop()], axis=-1),temb=temb, out_ch=ch * ch_mult[i_level])
# pop 10
h = resnet_block(tf.concat([h, hs.pop()], axis=-1),temb=temb, out_ch=ch * ch_mult[i_level])
# pop 9
h = resnet_block(tf.concat([h, hs.pop()], axis=-1),temb=temb, out_ch=ch * ch_mult[i_level])


for i_level in reversed(range(num_resolutions)): # 4
    # Residual blocks for this resolution
    for i_block in range(num_res_blocks + 1): # 3
        h = resnet_block(tf.concat([h, hs.pop()], axis=-1),
                        temb=temb, out_ch=ch * ch_mult[i_level])
        if h.shape[1] in attn_resolutions:
        h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
    # Upsample
    if i_level != 0:
        h = upsample(h, name='upsample', with_conv=resamp_with_conv)
assert not hs

#%%
def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  timesteps: array of ints describing the timestep each "picture" of the batch is perturbed to.\n
  timesteps.shape = B\n
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  \n
  Credit to DDPM (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90)
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = jnp.log(10000) / (half_dim - 1)
  emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.int32) * -emb)
  emb = jnp.int32(timesteps)[:, None] * emb[None, :]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = jnp.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

get_timestep_embedding(jnp.arange(3),embedding_dim=16)
#%%

import sys
sys.path.append("/media/sf_Bsc-Diffusion")
# need ends

from utils.utils import get_hydra_config
cfg = get_hydra_config()
# print(cfg.model)

mpa = cfg.model.parameters.model_parameter_association

mpa[0]
