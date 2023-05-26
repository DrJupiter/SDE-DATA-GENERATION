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
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

######################## Basic building blocks ########################

from models.ddpm.building_blocks.ddpm_functional_blocks import get_resnet_ff, get_attention, get_timestep_embedding, get_conv, get_down, get_down_attn, get_up, get_up_attn, get_text_embedding, get_text_data_embedding
from models.ddpm_classifier.building_blocks.ddpm_functional_blocks import get_lin_log_softmax

######################## MODEL ########################


def get_ddpm_unet(cfg, key, inference=False):

    data_c = cfg.dataset.shape[-1]
    c_s = cfg.model.parameters.Channel_sizes
    f_s = cfg.model.hyperparameters.scaling_factors

    # get new kets
    key, subkey = random.split(key,2)

    # get time embedding func and params
    #apply_timestep_embedding, p_embed = get_timestep_embedding(cfg, key) # B -> (cool stuff) -> B x Embedding_dim -> B x cfg.dim

    # Text embedding which is added to time
    #apply_text_embedding, p_text_embed = get_text_embedding(cfg, key) 

    # Text embedding which is added to the data
    #apply_text_embedding_data, p_text_data =get_text_data_embedding(cfg, key)

    # get model funcs and params
    conv1, p_c1, inf_conv1 =       get_conv(cfg, key, data_c, c_s[0], first=True)

    down1, p_d1, inf_down1 =       get_down(cfg, key, c_s[0], c_s[0], factor = f_s[0])
    down2_attn, p_da2, inf_down2_attn = get_down_attn(cfg, key, c_s[0], c_s[1], factor = f_s[1])
    down3, p_d3, inf_down3 =       get_down(cfg, key, c_s[1], c_s[2], factor = f_s[2])
    down4, p_d4, inf_down4 =       get_down(cfg, key, c_s[2], c_s[3], factor = f_s[3])

    r1, p_mr1, inf_r1 = get_resnet_ff(cfg, key, c_s[3], c_s[3])
    a1, p_ma2, inf_a1 = get_attention(cfg, key, c_s[3], c_s[3])
    r2, p_mr3, inf_r2 = get_resnet_ff(cfg, key, c_s[3], c_s[3])

    up1, p_u1, inf_up1 =         get_up(cfg, key, c_s[3], c_s[2], residual_C = [c_s[3],c_s[3],c_s[2]], factor = f_s[2])
    up2, p_u2, inf_up2 =         get_up(cfg, key, c_s[2], c_s[2], residual_C = [c_s[2],c_s[2],c_s[1]], factor = f_s[1])
    up_attn3, p_ua3, inf_up_attn3 =   get_up_attn(cfg, key, c_s[2], c_s[1], residual_C = [c_s[1],c_s[1],c_s[0]], factor = f_s[0])
    up4, p_u4, inf_up4 =         get_up(cfg, key, c_s[1], c_s[0], residual_C = [c_s[0],c_s[0],c_s[0]], factor = f_s[3])

    conv2, p_c2, inf_conv2 =       get_conv(cfg, key, c_s[0], data_c)

    lin_log_softmax, p_softmax = get_lin_log_softmax(cfg, key)

    # define all the aprams in a dict
    params = {"p_d1":p_d1, "p_da2":p_da2, "p_d3":p_d3, "p_d4":p_d4,  # down
              "p_u1":p_u1, "p_u2":p_u2, "p_ua3":p_ua3, "p_u4":p_u4,  # up
              "p_mr1":p_mr1, "p_ma2":p_ma2, "p_mr3":p_mr3, # middle
              "p_c1":p_c1, "p_c2":p_c2, # conv
              #"p_embed": p_embed,# time embedding
              #"p_text_embed": p_text_embed, # text embedding,
              #"p_text_embed_data": p_text_data, # text data embedding
              "p_softmax" : p_softmax
              }  

    if cfg.model.sharding:
        print("Sharding")
        params = get_model_sharding(cfg)(params)

    data_shape = [cfg.dataset.shape[0], cfg.dataset.shape[1]+cfg.dataset.padding*2, cfg.dataset.shape[2]+cfg.dataset.padding*2, cfg.dataset.shape[-1]]
    
    # forward ini:
    def ddpm_unet(x_in, timesteps, text_embedding, params, key):

        # Keep this the first line to ensure correct shapes for the SDE solver
        x_in_shape = x_in.shape

        # apply text guidance embedding

        #x_in = apply_text_embedding_data(x_in, text_embedding, params["p_text_embed_data"])

        # Transform input into the image shape
        x_in = x_in.reshape(data_shape)

        # Split key to preserve randomness
        key, *subkey = random.split(key,12) # TODO: change 2 to 29 and add a * in front of subkey

        # Create the embedding given the timesteps
        #embed = apply_timestep_embedding(timesteps*999, params["p_embed"])
        #text_embed = apply_text_embedding(text_embedding, params["p_text_embed"])
        #embed = embed + text_embed
        #embed = embed * 0
        embed = jnp.zeros((x_in_shape[0],cfg.model.hyperparameters.time_embedding_dims))

        # Apply model
        x_32_0 = conv1(x_in, params["p_c1"])

        x_32_1, x_32_2, x_16_0 = down1(x_32_0, embed, params["p_d1"], subkey[0]) 
        x_16_1, x_16_2, x_8_0 = down2_attn(x_16_0, embed, params["p_da2"], subkey[1]) 
        x_8_1, x_8_2, x_4_0 = down3(x_8_0, embed, params["p_d3"], subkey[2]) 
        x_4_1, x_4_2, x = down4(x_4_0, embed, params["p_d4"], subkey[3]) 

        x = r1(x, embed, params["p_mr1"], subkey[4])
        x = a1(x, embed, params["p_ma2"], subkey[5])
        x = r2(x, embed, params["p_mr3"], subkey[6])

        # x = jax.lax.with_sharding_constraint(x, sharding)

        x = up1(x, x_4_2, x_4_1, x_4_0, embed, params["p_u1"], subkey[7]) 
        x = up2(x, x_8_2, x_8_1, x_8_0, embed, params["p_u2"], subkey[8])
        x = up_attn3(x, x_16_2, x_16_1, x_16_0, embed, params["p_ua3"], subkey[9]) 
        x = up4(x, x_32_2, x_32_1, x_32_0, embed, params["p_u4"], subkey[10]) 

        x = conv2(x, params["p_c2"])

        # x = jax.lax.with_sharding_constraint(x, sharding.reshape(1,1,n_devices,1))

        # return to shape loss can take (input shape)
        x_out = x.reshape(x_in_shape) 

        # Transform to the size of labels and apply log_softmax
        x_out = lin_log_softmax(x_out, params["p_softmax"])
        return x_out

    def inf_ddpm_unet(x_in, timesteps, text_embedding, params, key):

        # Keep this the first line to ensure correct shapes for the SDE solver
        x_in_shape = x_in.shape

        # apply text guidance embedding

        #x_in = apply_text_embedding_data(x_in, text_embedding, params["p_text_embed_data"])

        #data_shape = jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0])
        # Transform input into the image shape
        x_in = x_in.reshape(data_shape)

        # Split key to preserve randomness
        key, *subkey = random.split(key,12) # TODO: change 2 to 29 and add a * in front of subkey

        # Create the embedding given the timesteps
        embed = jnp.zeros((x_in_shape[0],cfg.model.hyperparameters.time_embedding_dims))

        # Apply model
        x_32_0 = inf_conv1(x_in, params["p_c1"])

        x_32_1, x_32_2, x_16_0 = inf_down1(x_32_0, embed, params["p_d1"], subkey[0]) 
        x_16_1, x_16_2, x_8_0 = inf_down2_attn(x_16_0, embed, params["p_da2"], subkey[1])
        x_8_1, x_8_2, x_4_0 = inf_down3(x_8_0, embed, params["p_d3"], subkey[2]) 
        x_4_1, x_4_2, x = inf_down4(x_4_0, embed, params["p_d4"], subkey[3]) 

        x = inf_r1(x, embed, params["p_mr1"], subkey[4])
        x = inf_a1(x, embed, params["p_ma2"], subkey[5])
        x = inf_r2(x, embed, params["p_mr3"], subkey[6])

        # x = jax.lax.with_sharding_constraint(x, sharding)

        x = inf_up1(x, x_4_2, x_4_1, x_4_0, embed, params["p_u1"], subkey[7])
        x = inf_up2(x, x_8_2, x_8_1, x_8_0, embed, params["p_u2"], subkey[8]) 
        x = inf_up_attn3(x, x_16_2, x_16_1, x_16_0, embed, params["p_ua3"], subkey[9]) 
        x = inf_up4(x, x_32_2, x_32_1, x_32_0, embed, params["p_u4"], subkey[10])

        x = inf_conv2(x, params["p_c2"])

        # x = jax.lax.with_sharding_constraint(x, sharding.reshape(1,1,n_devices,1))

        # return to shape loss can take (input shape)
        x_out = x.reshape(x_in_shape) 

        # Transform to the size of labels and apply log_softmax
        x_out = lin_log_softmax(x_out, params["p_softmax"])
        return x_out

    # return func and parameters
    return params, ddpm_unet, inf_ddpm_unet
