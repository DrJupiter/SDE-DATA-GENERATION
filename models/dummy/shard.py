from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, NamedSharding, PartitionSpec
from jax import device_count, device_put 
import jax
import utils.sharding

def shard_parameters(cfg,parameters):
    print("Sharding model")
    n = device_count()
    names, mesh = utils.sharding.get_sharding(cfg)

    sharding = PositionalSharding(mesh_utils.create_device_mesh((n,)))
    for i in range(len(parameters)):
        parameters[i] = device_put(parameters[i], sharding.reshape(n, 1))
        jax.debug.visualize_array_sharding(parameters[i])
    return parameters