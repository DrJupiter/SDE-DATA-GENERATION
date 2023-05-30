# #%%
# import jax.numpy as jnp
# from jax import lax, jit
# in_dim = 2
# img1 = jnp.arange(3*in_dim*32*32).reshape(3,in_dim,32,32)
# k1 = jnp.arange(10*in_dim*3*3).reshape(10,in_dim,3,3)

# e = lax.conv_general_dilated( 
#             lhs = img1,   
#             rhs = k1, 
#             window_strides = [1,1], 
#             padding = 'same',
#             dimension_numbers = ('NCHW', 'OIHW', 'NCHW')
#             )

# img2 = img1.transpose(0,2,3,1)
# k2 = k1.transpose(2,3,1,0)
# e2 = lax.conv_general_dilated( 
#             lhs = img2,    
#             rhs = k2, 
#             window_strides = [1,1], 
#             padding = 'same',
#             dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
#             )

# # print("IMG1\n",img1)
# # print("IMG2\n",img2)
# print(img2.shape)
# print(k2.shape)
# print(jnp.sum(k1),jnp.sum(k2),"\n",jnp.sum(img1),jnp.sum(img2))
# print("NCHW:",jnp.sum(e),"\nNHWC",jnp.sum(e2))
# #%%

# img1.shape
# # 1,3,3,in_dim
# print(img1.transpose(0,2,3,1))
# img2

# #%%
# import equinox as eqx
# import jax
# import jax.numpy as jnp
# import jax.random as jr

# key = jr.PRNGKey(0)
# mkey, dkey = jr.split(key)
# model = eqx.nn.Sequential([
#     eqx.experimental.BatchNorm(input_size=4, axis_name="batch"),
# ])

# x = jr.normal(dkey, (2,32,32,4))
# jax.vmap(model, axis_name="batch")(x.transpose(0,3,2,1)).transpose(0,3,2,1).shape
# # BatchNorm will automatically update its running statistics internally.
# # %%

# import random
# from typing import Tuple

# import optax
# import jax.numpy as jnp
# import jax
# import numpy as np

# BATCH_SIZE = 5
# NUM_TRAIN_STEPS = 1_000
# RAW_TRAINING_DATA = np.random.randint(255, size=(NUM_TRAIN_STEPS, BATCH_SIZE, 1))

# TRAINING_DATA = np.unpackbits(RAW_TRAINING_DATA.astype(np.uint8), axis=-1)
# LABELS = jax.nn.one_hot(RAW_TRAINING_DATA % 2, 2).astype(jnp.float32).reshape(NUM_TRAIN_STEPS, BATCH_SIZE, 2)


# # %%
# initial_params = {
#     'hidden': jax.random.normal(shape=[8, 32], key=jax.random.PRNGKey(0)),
#     'output': jax.random.normal(shape=[32, 2], key=jax.random.PRNGKey(1)),
# }


# def net(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#   x = jnp.dot(x, params['hidden'])
#   x = jax.nn.relu(x)
#   x = jnp.dot(x, params['output'])
#   return x


# def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
#   y_hat = net(batch, params)

#   # optax also provides a number of common loss functions.
#   loss_value = optax.sigmoid_binary_cross_entropy(y_hat, labels).sum(axis=-1)

#   return loss_value.mean()

# # %%
# def fit(params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
#   opt_state = optimizer.init(params)

#   @jax.jit
#   def step(params, opt_state, batch, labels):
#     loss_value, grads = jax.value_and_grad(jit(loss))(params, batch, labels)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, loss_value

#   for i, (batch, labels) in enumerate(zip(TRAINING_DATA, LABELS)):
#     params, opt_state, loss_value = step(params, opt_state, batch, labels)
#     if i % 100 == 0:
#       print(f'step {i}, loss: {loss_value}')

#   return params

# # Finally, we can fit our parametrized function using the Adam optimizer
# # provided by optax.
# optimizer = optax.adam(learning_rate=1e-2)
# params = fit(initial_params, optimizer)

# # %%
# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# import jax
# import equinox as eqx


# class tst(eqx.Module):
#     def __init__(self):
#       self.adwds=2

# tst()

# #%%
# import jax.numpy as jnp
# from jax import vmap

# def fisk(x):
#   return x**2


# print(vmap(fisk,(0),0)(X))

# # %%

# def naive_downsample_2d(x, factor=2):
#   _N, H, W, C = x.shape
#   x = jnp.reshape(x, [-1, H // factor, factor, W // factor, factor, C])
#   return jnp.mean(x, axis=[2, 4])

# X = jnp.ones((3,32,32,2))*2

# naive_downsample_2d(X)

#%%
import jax
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
print(jax.devices())

#%%

# TODO: REMOVE THIS
import jax
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
print(jax.devices())

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,))).reshape(4,1)

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key,2)

x = jax.random.normal(key1, (8192, 8192))
w = jax.random.normal(key2, (8192, 8192))
# and use jax.device_put to distribute it across devices:
xs = jax.device_put(x, sharding)
ws = jax.device_put(x, sharding.rehsape(-1,1))

jax.debug.visualize_array_sharding(xs)
jax.debug.visualize_array_sharding(ws)



#%%
jax.numpy.matmul(x,w)


# %%
jax.numpy.matmul(x,ws)

#%%

jax.numpy.matmul(xs,w)
#%%
jax.numpy.matmul(xs,ws)

# %%
import jax
# initialization of key and two variables x and w
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key,2)
x = jax.random.normal(key1, (512,512))
w = jax.random.normal(key2, (512,512))

# create sharding
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
# using 4 devices
sharding = PositionalSharding(mesh_utils.create_device_mesh((4,))).reshape(4,1)

# shard x and w
x = jax.device_put(x,sharding)
w = jax.device_put(w,sharding)

# perform some computation
y = jax.numpy.matmul(x,w)
jax.debug.visualize_array_sharding(x)

#%%
import jax
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
n_devices = len(jax.devices())
print("n: ",n_devices)

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,))).reshape(n_devices,1)

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key,2)

C = 8192

x = jax.device_put(jax.random.normal(key1, (C, C)), sharding)
w = jax.device_put(jax.random.normal(key2, (C, C)), sharding.reshape(1,-1))

jax.debug.visualize_array_sharding(x)
jax.debug.visualize_array_sharding(w)

# %%

import jax
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
print(jax.devices())

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax.numpy as jnp

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,))).reshape(1,4)

x = jax.device_put(jnp.ones((4,16*16*32)),sharding)

print(x.sharding)

x = x.reshape(4,16,16,32)

print(x.sharding)
