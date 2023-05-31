from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, NamedSharding
from jax import device_count, device_put 
import jax
import utils.sharding

def shard_parameters(cfg,parameters):
    print("Sharding model")
    n = device_count()
    shard_names, mesh = utils.sharding.get_sharding(cfg)
    named_sharding = NamedSharding(mesh, PartitionSpec(shard_names))

    #sharding = PositionalSharding(mesh_utils.create_device_mesh((n,)))
    for i in range(len(parameters)):
        parameters[i] = device_put(parameters[i], named_sharding)
        jax.debug.visualize_array_sharding(parameters[i])
    return parameters