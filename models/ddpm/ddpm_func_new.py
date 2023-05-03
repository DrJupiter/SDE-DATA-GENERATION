# stop prelocation of memory
# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax
import jax

# sharding
from utils.utils import get_model_sharding

######################## Basic building blocks ########################

from models.ddpm.building_blocks.ddpm_func_new import get_resnet_ff, get_attention, get_timestep_embedding, get_conv, get_down, get_down_attn, get_up, get_up_attn

######################## MODEL ########################


def get_ddpm_unet(cfg, key, inference=False):

    data_c = cfg.dataset.shape[-1]
    c_s = cfg.model.parameters.Channel_sizes

    # get new kets
    key, subkey = random.split(key,2)

    # get time embedding func and params
    apply_timestep_embedding, p_embed = get_timestep_embedding(cfg, key, embedding_dim=c_s[0])

    # get model funcs and params
    conv1, p_c1 =       get_conv(cfg, key, data_c, c_s[0])

    down1, p_d1 =       get_down(cfg, key, c_s[0], c_s[0], inference=inference)
    down2_attn, p_da2 = get_down_attn(cfg, key, c_s[0], c_s[1], inference=inference)
    down3, p_d3 =       get_down(cfg, key, c_s[1], c_s[2], inference=inference)
    down4, p_d4 =       get_down(cfg, key, c_s[2], c_s[3], inference=inference)

    r1, p_mr1 = get_resnet_ff(cfg, key, c_s[3], c_s[3], inference=inference)
    a1, p_ma2 = get_attention(cfg, key, c_s[3], c_s[3], inference=inference)
    r2, p_mr3 = get_resnet_ff(cfg, key, c_s[3], c_s[3], inference=inference)

    up1, p_u1 =         get_up(cfg, key, c_s[3], c_s[2], residual_C = [c_s[3],c_s[3],c_s[2]], inference=inference)
    up2, p_u2 =         get_up(cfg, key, c_s[2], c_s[2], residual_C = [c_s[2],c_s[2],c_s[1]], inference=inference)
    up_attn3, p_ua3 =   get_up_attn(cfg, key, c_s[2], c_s[1], residual_C = [c_s[1],c_s[1],c_s[0]], inference=inference)
    up4, p_u4 =         get_up(cfg, key, c_s[1], c_s[0], residual_C = [c_s[0],c_s[0],c_s[0]], inference=inference)

    conv2, p_c2 =       get_conv(cfg, key, c_s[0], data_c)

    # define all the aprams in a dict
    params = {"p_d1":p_d1, "p_da2":p_da2, "p_d3":p_d3, "p_d4":p_d4,  # down
              "p_u1":p_u1, "p_u2":p_u2, "p_ua3":p_ua3, "p_u4":p_u4,  # up
              "p_mr1":p_mr1, "p_ma2":p_ma2, "p_mr3":p_mr3, # middle
              "p_c1":p_c1, "p_c2":p_c2, # conv
              "p_embed": p_embed}  # time embedding

    if cfg.model.sharding:
        print("Sharding")
        params = get_model_sharding(cfg)(params)

    # forward ini:
    def ddpm_unet(x_in, timesteps, params, key):

        # Transform input into the image shape
        x_in_shape = x_in.shape
        x_in = x_in.reshape(cfg.dataset.shape)

        # Split key to preserve randomness
        key, subkey = random.split(key,2) # TODO: change 2 to 29 and add a * in front of subkey

        # Create the embedding given the timesteps
        embed = apply_timestep_embedding(timesteps*999, params["p_embed"])

        # Apply model
        x_32_0 = conv1(x_in, params["p_c1"])

        x_32_1, x_32_2, x_16_0 = down1(x_32_0, embed, params["p_d1"], subkey, factor=1) # factor = 2
        x_16_1, x_16_2, x_8_0 = down2_attn(x_16_0, embed, params["p_da2"], subkey, factor=2)
        x_8_1, x_8_2, x_4_0 = down3(x_8_0, embed, params["p_d3"], subkey, factor=2)
        x_4_1, x_4_2, x = down4(x_4_0, embed, params["p_d4"], subkey, factor=1)

        x = r1(x, embed, params["p_mr1"], subkey)
        x = a1(x, embed, params["p_ma2"], subkey)
        x = r2(x, embed, params["p_mr3"], subkey)

        x = up1(x, x_4_2, x_4_1, x_4_0, embed, params["p_u1"], subkey, factor=1) # factor = 2
        x = up2(x, x_8_2, x_8_1, x_8_0, embed, params["p_u2"], subkey, factor=2)
        x = up_attn3(x, x_16_2, x_16_1, x_16_0, embed, params["p_ua3"], subkey, factor=2)
        x = up4(x, x_32_2, x_32_1, x_32_0, embed, params["p_u4"], subkey, factor=1)

        x = conv2(x, params["p_c2"])

        # return to shape loss can take (input shape)
        x_out = x.reshape(x_in_shape) 

        return x_out

    # return func and parameters
    return params, ddpm_unet
