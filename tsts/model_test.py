## Adding path (for some it is needed to import packages)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


# Stop jax from taking up 90% of GPU vram
# import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.5'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'



import jax
import jax.numpy as jnp


# from jax import config
# config.update("jax_disable_jit", True)


from models.model import get_model 
from models.ddpm.building_blocks.ddpm_func_new import get_resnet_ff, get_attention, get_down_attn, get_up_attn

from time import sleep

from utils.utility import get_hydra_config
    
from loss.loss import get_loss

cfg = get_hydra_config()

print("num devices used = ",len(jax.devices()))

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from data.dataload import dataload 
# Load train and test sets



# Get randomness key
key = jax.random.PRNGKey(22)
key, subkey = jax.random.split(key)


# try submodels
# x = jnp.ones(1,32,32,4)
# embed = jnp.ones(1,32,32,4)

sharding = PositionalSharding(mesh_utils.create_device_mesh((len(jax.devices()),1)))


x = jax.device_put(jnp.ones((4,4,8,4),dtype=jnp.float32)*10,sharding.reshape(len(jax.devices()),1,1,1))
embed = jax.device_put(jnp.ones((4,128),dtype=jnp.float32)*10,sharding.reshape(len(jax.devices()),1))

resnet, paramsr = get_resnet_ff(cfg, key, 4, 4)
attn, paramsa = get_attention(cfg, key, 4, 4)
down, paramsd = get_down_attn(cfg, key, 4, 4)
up, paramsu = get_up_attn(cfg, key, 4, 4, residual_C=[4,4,4])

def g_resnet(x, embed, paramsr, key):
    out = resnet(x, embed, paramsr, key)
    return jnp.sum(out)

def g_attn(x, embed, paramsr, key):
    out = attn(x, embed, paramsr, key)
    return jnp.sum(out)

def g_down(x, embed, paramsd, key, factor=1):
    out = down(x, embed, paramsd, key, factor=factor)
    return jnp.sum(out[-1])

def g_up(x, embed, paramsd, key, factor=1):
    out = up(x, x, x, x, embed, paramsd, key, factor=factor)
    return jnp.sum(out)

gg_resnet = jax.grad(g_resnet,2)
gg_attn = jax.grad(g_attn,2)
gg_down = jax.grad(g_down,2)
gg_up = jax.grad(g_up,2)



outr = resnet(x, embed, paramsr, key)
print("resnet",outr.shape,"vals=",outr[0][0][0][0])

outa = attn(x, embed, paramsa, key)
print("attn",outa.shape,"vals=",outa[0][0][0][0])

outd = down(x, embed, paramsd, key, factor=1)
print("down attn",outd[-1].shape,"vals=",outd[0][0][0][0])

outu = up(x, x, x, x, embed, paramsu, key, factor=1)
print("up attn",outu.shape,"vals=",outu[0][0][0][0])


goutr = gg_resnet(x, embed, paramsr, key)
print("grad resnet",goutr.keys())#,"vals=",goutr)

gouta = gg_attn(x, embed, paramsa, key)
print("grad attn",gouta.keys())#,"vals=",gouta[0][0][0][0])

goutd = gg_down(x, embed, paramsd, key, factor=1)
print("grad down attn",goutd.keys())#),"vals=",goutd[0][0][0][0])

goutu = gg_up(x, embed, paramsu, key, factor=1)
print("grad up attn",goutu.keys())#,"vals=",goutu[0][0][0][0])

# # print("start sleeping")
# # sleep(2)
# # print("end sleeping")


# # Get model forward call and its parameters
# model_parameters, model_call, infmodel = get_model(cfg, key = subkey) # model_call(x_in, timesteps, parameters)


# # print("start sleeping")
# # sleep(2)
# # print("end sleeping")

# train_dataset, test_dataset = dataload(cfg) 

# # from sde.sde import get_sde
# # SDE = get_sde(cfg)



# loss_fn = get_loss(cfg) # loss_fn(func, function_parameters, data, perturbed_data, time, key)
# grad_fn = jax.grad(loss_fn,1)

# for i, (data,label) in enumerate(train_dataset):
#     timesteps = 999*jax.random.uniform(key, (data.shape[0],), minval=1e-5, maxval=1)
#     # perturbed_data = SDE.sample(timesteps, data, key)


#     print(data.shape)
#     data = jax.device_put(data ,sharding.reshape((1,len(jax.devices()))))
#     out = model_call(data, jnp.ones(1), model_parameters, key)
#     print(out)
#     grads = grad_fn(model_call, model_parameters, data, data, timesteps, key)
#     print(grads.keys())
#     break



# print("start sleeping")
# sleep(2)
# print("end sleeping")


# print("gradding")

# print("done")
# print("start sleeping")
# sleep(2)
# print("end sleeping")
