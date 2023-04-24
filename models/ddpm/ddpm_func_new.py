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
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

######################## Basic building blocks ########################

from models.ddpm.building_blocks.ddpm_func_new import get_resnet_ff, get_attention, get_timestep_embedding, get_conv, get_down, get_down_attn, get_up, get_up_attn

######################## MODEL ########################


def get_ddpm_unet(cfg, key):

    data_c = cfg.dataset.shape[-1]
    c_s = cfg.model.parameters.Channel_sizes

    # get new kets
    key, subkey = random.split(key,2)

    # if cfg.model.hyperparameters.sharding:
    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,1)))

    # get time embedding func and params
    apply_timestep_embedding, p_embed = get_timestep_embedding(cfg, key, embedding_dim=c_s[0], sharding = sharding)

    # get model funcs and params
    conv1, p_c1 =       get_conv(cfg, key, data_c, c_s[0], sharding.reshape((1,1,1,len(jax.devices()))))

    down1, p_d1 =       get_down(cfg, key, c_s[0], c_s[0], sharding)
    down2_attn, p_da2 = get_down_attn(cfg, key, c_s[0], c_s[1], sharding)
    down3, p_d3 =       get_down(cfg, key, c_s[1], c_s[2], sharding)
    down4, p_d4 =       get_down(cfg, key, c_s[2], c_s[3], sharding)

    r1, p_mr1 = get_resnet_ff(cfg, key, c_s[3], c_s[3], sharding)
    a1, p_ma2 = get_attention(cfg, key, c_s[3], c_s[3], sharding)
    r2, p_mr3 = get_resnet_ff(cfg, key, c_s[3], c_s[3], sharding)

    up1, p_u1 =         get_up(cfg, key, c_s[3], c_s[2], residual_C = [c_s[3],c_s[3],c_s[2]], sharding = sharding)
    up2, p_u2 =         get_up(cfg, key, c_s[2], c_s[2], residual_C = [c_s[2],c_s[2],c_s[1]], sharding = sharding)
    up_attn3, p_ua3 =   get_up_attn(cfg, key, c_s[2], c_s[1], residual_C = [c_s[1],c_s[1],c_s[0]], sharding = sharding)
    up4, p_u4 =         get_up(cfg, key, c_s[1], c_s[0], residual_C = [c_s[0],c_s[0],c_s[0]], sharding = sharding)

    conv2, p_c2 =       get_conv(cfg, key, c_s[0], data_c, sharding.reshape((1,1,len(jax.devices()),1)))

    # define all the aprams in a dict
    params = {"p_d1":p_d1, "p_da2":p_da2, "p_d3":p_d3, "p_d4":p_d4,  # down
              "p_u1":p_u1, "p_u2":p_u2, "p_ua3":p_ua3, "p_u4":p_u4,  # up
              "p_mr1":p_mr1, "p_ma2":p_ma2, "p_mr3":p_mr3, # middle
              "p_c1":p_c1, "p_c2":p_c2, # conv
              "p_embed": p_embed}  # time embedding

    # forward ini:
    def ddpm_unet(x_in, timesteps, params, key):

        # Transform input into the image shape
        x_in_shape = x_in.shape
        x_in = x_in.reshape(cfg.dataset.shape)

        # Split key to preserve randomness
        key, subkey = random.split(key,2) # TODO: change 2 to 29 and add a * in front of subkey

        # Create the embedding given the timesteps
        embed = apply_timestep_embedding(timesteps, params["p_embed"])

        # Apply model
        x_32_0 = conv1(x_in, params["p_c1"])

        x_32_1, x_32_2, x_16_0 = down1(x_32_0, embed, params["p_d1"], subkey, factor=2)
        x_16_1, x_16_2, x_8_0 = down2_attn(x_16_0, embed, params["p_da2"], subkey, factor=2)
        x_8_1, x_8_2, x_4_0 = down3(x_8_0, embed, params["p_d3"], subkey, factor=2)
        x_4_1, x_4_2, x = down4(x_4_0, embed, params["p_d4"], subkey, factor=1)

        x = r1(x, embed, params["p_mr1"], subkey)
        x = a1(x, embed, params["p_ma2"], subkey)
        x = r2(x, embed, params["p_mr3"], subkey)

        x = up1(x, x_4_2, x_4_1, x_4_0, embed, params["p_u1"], subkey, factor=2)
        x = up2(x, x_8_2, x_8_1, x_8_0, embed, params["p_u2"], subkey, factor=2)
        x = up_attn3(x, x_16_2, x_16_1, x_16_0, embed, params["p_ua3"], subkey, factor=2)
        x = up4(x, x_32_2, x_32_1, x_32_0, embed, params["p_u4"], subkey, factor=1)

        x = conv2(x, params["p_c2"])

        # return to shape loss can take (input shape)
        x_out = x.reshape(x_in_shape) 

        return x_out

    # return func and parameters
    return params, ddpm_unet


def get_ddpm_unet2(cfg):

    def ddpm_unet(x_in, timesteps, parameters, key):

        upsampling_factor = cfg.model.parameters.upsampling_factor
        maxpool_factor = cfg.model.parameters.downsampling_factor

        embedding_dims = cfg.model.parameters.time_embedding_dims

        param_asso = jnp.array(cfg.model.parameters.model_parameter_association)


        # Transform input into the image shape
        x_in_shape = x_in.shape
        x_in = x_in.reshape(cfg.dataset.shape)


        # Split key to preserve randomness
        key, subkey = random.split(key,2) # TODO: change 2 to 29 and add a * in front of subkey

        # Get parameters for Timestep embedding
        em_w1 = parameters[2][0][-2]
        em_w2 = parameters[2][0][-1]
        em_b1 = parameters[2][1][-2]
        em_b2 = parameters[2][1][-1]

        # Create the embedding given the timesteps
        embedding = get_timestep_embedding(timesteps, embedding_dim = embedding_dims) # embedding -> dense -> nonlin -> dense (Shape = Bx512)
        embedding = jnp.matmul(embedding,em_w1)+em_b1 # 128 -> 512
        embedding = nn.relu(embedding)
        embedding = jnp.matmul(embedding,em_w2)+em_b2 # 512 -> 512

        # Perform main pass:
        ## down
        x_32_0 = lax.conv_general_dilated( 
                lhs = x_in,    
                rhs = parameters[0][0], # kernel is the conv [0] and the first entry i this [0]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC') # input and parameter shapes
                )
        
        ### DOWN ###
        # Down ResNet 1
        x_32_1 = resnet_ff(x_32_0,    embedding, parameters,subkey = subkey, local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=1)
        x_32_2 = resnet_ff(x_32_1,    embedding, parameters,subkey = subkey, local_num_shift = 2, cfg=cfg, param_asso= param_asso,sub_model_num=1)
        # maxpool (changes shape)
        x_16_0 = naive_downsample_2d(x_32_2, factor = maxpool_factor)

        # Down ResNet Attention 2
        x = resnet_ff(x_16_0,    embedding, parameters, subkey = subkey, local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=2)
        x_16_1 = attention(x,    embedding, parameters, subkey = subkey, local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=2)
        x = resnet_ff(x_16_1,    embedding, parameters, subkey = subkey, local_num_shift = 2, cfg=cfg, param_asso=param_asso, sub_model_num=2)
        x_16_2 = attention(x,    embedding, parameters, subkey = subkey, local_num_shift = 4, cfg=cfg, param_asso=param_asso, sub_model_num=2)
        # maxpool (changes shapes)
        # x_8_0 = naive_downsample_2d(x_16_2, factor = maxpool_factor)
        x = x_16_2
        
        # # Down ResNet 3
        # x_8_1 = resnet_ff(x_8_0,    embedding, parameters,subkey = subkey(), local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=3)
        # x_8_2 = resnet_ff(x_8_1,      embedding, parameters,subkey = subkey(), local_num_shift = 2, cfg=cfg, param_asso= param_asso,sub_model_num=3)
        # # maxpool (changes shape)
        # x_4_0 = naive_downsample_2d(x_8_2, factor = maxpool_factor)

        # # Down ResNet 4
        # x_4_1 = resnet_ff(x_4_0,    embedding, parameters,subkey = subkey(), local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=4)
        # x_4_2 = resnet_ff(x_4_1,      embedding, parameters,subkey = subkey(), local_num_shift = 2, cfg=cfg, param_asso= param_asso,sub_model_num=4)
        

        # ### MIDDLE ###
        # # ff 5
        # x = resnet_ff(            x_in = x_4_2, 
        #                           embedding = embedding, 
        #                           parameters = parameters, 
        #                           subkey = subkey, 
        #                           cfg = cfg.model, 
        #                           param_asso = param_asso, 
        #                           sub_model_num = 5, 
        #                           local_num_shift = 0, ) # 4x4 -> 4x4
        # # attn 6
        # x = attention(            x_in = x, 
        #                           embedding = embedding, 
        #                           parameters = parameters, 
        #                           subkey = subkey, 
        #                           cfg = cfg.model, 
        #                           param_asso = param_asso, 
        #                           sub_model_num = 6, 
        #                           local_num_shift = 0, ) # 4x4 -> 4x4
        # # ff 7
        # x = resnet_ff(            x_in = x, 
        #                           embedding = embedding, 
        #                           parameters = parameters, 
        #                           subkey = subkey, 
        #                           cfg = cfg.model, 
        #                           param_asso = param_asso, 
        #                           sub_model_num = 7, 
        #                           local_num_shift = 0, ) # 4x4 -> 4x4   C_out = 512


        # # Up ResNet 8
        # x = resnet_ff(jnp.concatenate([x,x_4_2],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=8, local_num_shift = 0)
        # x = resnet_ff(jnp.concatenate([x,x_4_1],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=8, local_num_shift = 2)
        # x = resnet_ff(jnp.concatenate([x,x_4_0],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=8, local_num_shift = 4)
        # # Upsample
        # x = upsample2d(           x, factor=upsampling_factor) # 16x16 -> 32x32

        # # Up ResNet 9
        # x = resnet_ff(jnp.concatenate([x,x_8_2],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=9, local_num_shift = 0)
        # x = resnet_ff(jnp.concatenate([x,x_8_1],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=9, local_num_shift = 2)
        # x = resnet_ff(jnp.concatenate([x,x_8_0],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=9, local_num_shift = 4)
        # # Upsample
        # x = upsample2d(           x, factor=upsampling_factor) # 16x16 -> 32x32
        # print(x)


        # Up ResNet Attn 10
        x = resnet_ff(jnp.concatenate((x,x_16_2),axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=10, local_num_shift = 0)
        x = attention(x, embedding, parameters, subkey = subkey[1], local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=10)
        x = resnet_ff(jnp.concatenate((x,x_16_1),axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=10, local_num_shift = 2)
        x = attention(x, embedding, parameters, subkey = subkey[3], local_num_shift = 4, cfg=cfg, param_asso=param_asso, sub_model_num=10)
        x = resnet_ff(jnp.concatenate((x,x_16_0),axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=10, local_num_shift = 4)
        x = attention(x, embedding, parameters, subkey = subkey[5], local_num_shift = 8, cfg=cfg, param_asso=param_asso, sub_model_num=10)
        # Upsample
        x = upsample2d(x, factor=upsampling_factor) # 16x16 -> 32x32


        # Up ResNet 11
        x = resnet_ff(jnp.concatenate([x,x_32_2],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=11, local_num_shift = 0)
        x = resnet_ff(jnp.concatenate([x,x_32_1],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=11, local_num_shift = 2)
        x = resnet_ff(jnp.concatenate([x,x_32_0],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=11, local_num_shift = 4)


        #3 end
        x = x # vmap(batchnorm_12,axis_name="batch")(u.transpose(0,3,2,1)).transpose(0,3,2,1)
        x = nn.relu(x)
        x = lax.conv_general_dilated( 
                lhs = x,    
                rhs = parameters[0][-1], # kernel is the conv [0] and the last entry of these [-1]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )

        # return to shape loss can take (input shape)
        x_out = x.reshape(x_in_shape) 

        return x_out


    return ddpm_unet 

