from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from jax import device_count, device_put 
import jax

def shard_parameters(cfg,parameters):
    print("Sharding model")
    n = device_count()
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n,)))
    for i in range(len(parameters)):
        parameters[i] = device_put(parameters[i], sharding.reshape(n, 1))
        jax.debug.visualize_array_sharding(parameters[i])
    return parameters