from jax.experimental import mesh_utils

from jax.sharding import Mesh, PartitionSpec, NamedSharding

import jax

import numpy as np

def get_sharding(cfg):
     shard_names = np.array(['B', 'H', 'W', 'C'])
     if cfg.loss.name == "implicit_score_matching":
          shard_distribution = (1, jax.device_count(), 1, 1)
     else:
          shard_distribution = (jax.device_count(), 1, 1, 1)

     mesh = Mesh(mesh_utils.create_device_mesh(shard_distribution), shard_names)
     primary = np.argmax(shard_distribution)
     secondary = np.where(shard_names != shard_names[primary])
     return (shard_names[primary], shard_names[secondary]), mesh

    #spec = PartitionSpec(('B',))
    #out_spec = PartitionSpec(('B', None, None, None))
    #named_sharding = NamedSharding(mesh, spec)