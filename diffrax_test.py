import functools as ft
import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.experimental.shard_map as shard_map
import jax.numpy as jnp
import jax.sharding as sharding
from diffrax import diffeqsolve, ODETerm, Tsit5

mesh = sharding.Mesh(mesh_utils.create_device_mesh((2,)), ["i"])
spec = sharding.PartitionSpec('i')

@jax.jit
@ft.partial(shard_map.shard_map, mesh=mesh, in_specs=spec, out_specs=spec, check_rep=False)
@jax.vmap
def run(y0):
    term = ODETerm(lambda t, y, args: -y)
    solver = Tsit5()
    t0 = 0
    t1 = 1
    dt0 = 0.1
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0)
    return sol.ys

sharding = sharding.NamedSharding(mesh, spec)
y0 = jnp.array([10.0, 10.0])
y0 = jax.device_put(y0, sharding)
out = run(y0)
print(out)
jax.debug.visualize_array_sharding(out)