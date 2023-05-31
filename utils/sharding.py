from jax.experimental import mesh_utils

from jax.sharding import Mesh, PartitionSpec, NamedSharding

import jax

def get_sharding(cfg):
    shard_names = ['B', 'D']
    if cfg.loss.name == "implicit_sm":
         mesh = Mesh(mesh_utils.create_device_mesh((1, len(jax.devices()))), shard_names)
    else:
         mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),1)), shard_names)
    
    return shard_names, mesh

    #spec = PartitionSpec(('B',))
    #out_spec = PartitionSpec(('B', None, None, None))
    #named_sharding = NamedSharding(mesh, spec)