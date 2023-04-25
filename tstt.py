import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.5'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
import jax

from jax import numpy as jnp
from jax import vmap

B = 8
C1 = 16
C2 = 12

a = jnp.arange(B*C1).reshape(B,C1)
b = jnp.arange(B*C1).reshape(B,C1)

batch_matmul = vmap(lambda a,b: jnp.matmul(a.T, b), (0, 0) , 0)

print(batch_matmul(a,b))




# sharding
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# TEST SHARDING TODO: REMOVE


sharding = PositionalSharding(mesh_utils.create_device_mesh((len(jax.devices())//2,2)))

a = jax.device_put(a,sharding.replicate(1))

jax.debug.visualize_array_sharding(a)

print(sharding)
#%%

import numpy as np

r = 2000/2001
var = 0
for i,x in enumerate(np.random.random(10000)):
    var = var*r+(x-0.5)/2001

    if i % 100==0:
        print(var)