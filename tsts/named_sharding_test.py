#%%
import jax
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
print(jax.devices())


#%%
from functools import partial
from jax.experimental.shard_map import shard_map

import jax.numpy as jnp
x = jnp.ones((16,16,16,32))
shard_names = ("B","H","W","C")


from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

P = PartitionSpec

devices = mesh_utils.create_device_mesh((8,1,1,1))
mesh = Mesh(devices, axis_names=shard_names)
x = jax.device_put(x, NamedSharding(mesh, P(*shard_names)))

x.sharding

w = jnp.ones((32,32))


jax.jit
@partial(shard_map, mesh=mesh, in_specs=P('B'), out_specs=P('B'))
@partial(args=)
def eins(x,w):
    print(x.shape,w.shape)
    return jnp.einsum("bhwc,Cc->bhwC",x,w)

o = eins(x,w)
print(o.shape)
o.sharding

# %%




from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

sharding = PositionalSharding(mesh_utils.create_device_mesh((4,))).reshape(1,4)

x = jax.device_put(jnp.ones((4,16*16*32)),sharding)

# %%
#%%
x = jnp.array([[3.]])
jnp.tile(x, (4, 2))
