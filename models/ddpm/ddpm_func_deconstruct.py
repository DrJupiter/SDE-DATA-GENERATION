# stop prelocation of memory
# import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax

# Equinox
# TODO: remove everything from the function itself, by defining another function that creates this function with the params already defined, as such we remove the repeatable unessary code
# TODO: replace all einsubs with matmul and the opposite
    # TODO: Deconstruct entire thing and manually inseart indexs for params
# TODO: try jit on grad as we for some reason dont do it now

# TODO: rewrite it all in equinox


######################## Basic building blocks ########################

from models.ddpm.building_blocks.ddpm_functions import down_resnet, down_resnet_attn, up_resnet, up_resnet_attn, upsample2d, naive_downsample_2d, get_timestep_embedding, resnet_ff, attention, get_resnet_ff, get_attention

######################## MODEL ########################


def get_ddpm_unet(cfg):

    # main model ini:
    upsampling_factor = cfg.model.parameters.upsampling_factor
    maxpool_factor = cfg.model.parameters.downsampling_factor
    embedding_dims = cfg.model.parameters.time_embedding_dims
    param_asso = jnp.array(cfg.model.parameters.model_parameter_association)

    resnet_ff_1_0 = get_resnet_ff(cfg, param_asso, sub_model_num=1, local_num_shift=0) 
    resnet_ff_1_1 = get_resnet_ff(cfg, param_asso, sub_model_num=1, local_num_shift=1) 

    resnet_ff_2_0 = get_resnet_ff(cfg, param_asso, sub_model_num=2, local_num_shift=0) 
    resnet_ff_2_1 = get_resnet_ff(cfg, param_asso, sub_model_num=2, local_num_shift=1) 
    attn_2_0 = get_attention(cfg, param_asso, sub_model_num=2, local_num_shift=0)
    attn_2_1 = get_attention(cfg, param_asso, sub_model_num=2, local_num_shift=1)

    resnet_ff_10_0 = get_resnet_ff(cfg, param_asso, sub_model_num=10, local_num_shift=0) 
    resnet_ff_10_1 = get_resnet_ff(cfg, param_asso, sub_model_num=10, local_num_shift=1) 
    resnet_ff_10_2 = get_resnet_ff(cfg, param_asso, sub_model_num=10, local_num_shift=2) 
    attn_10_0 = get_attention(cfg, param_asso, sub_model_num=10, local_num_shift=0)
    attn_10_1 = get_attention(cfg, param_asso, sub_model_num=10, local_num_shift=1)
    attn_10_2 = get_attention(cfg, param_asso, sub_model_num=10, local_num_shift=2)

    resnet_ff_11_0 = get_resnet_ff(cfg, param_asso, sub_model_num=11, local_num_shift=0) 
    resnet_ff_11_1 = get_resnet_ff(cfg, param_asso, sub_model_num=11, local_num_shift=1) 
    resnet_ff_11_2 = get_resnet_ff(cfg, param_asso, sub_model_num=11, local_num_shift=2) 

    # submodel ini:
    def ddpm_unet(x_in, timesteps, parameters, key):

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
        x_32_1 = resnet_ff_1_0(x_32_0,    embedding, parameters,subkey = subkey)
        x_32_2 = resnet_ff_1_1(x_32_1,    embedding, parameters,subkey = subkey)
        # maxpool (changes shape)
        x_16_0 = naive_downsample_2d(x_32_2, factor = maxpool_factor)

        # Down ResNet Attention 2
        x = resnet_ff_2_0(x_16_0,    embedding, parameters, subkey = subkey)
        x_16_1 = attn_2_0(x,    embedding, parameters, subkey = subkey)
        x = resnet_ff_2_1(x_16_1,    embedding, parameters, subkey = subkey)
        x_16_2 = attn_2_1(x,    embedding, parameters, subkey = subkey)
        # maxpool (changes shapes)
        x_8_0 = naive_downsample_2d(x_16_2, factor = maxpool_factor)
        #x = x_16_2+0.0
        
        # Down ResNet 3
        x_8_1 = resnet_ff(x_8_0,    embedding, parameters,subkey = subkey, local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=3)
        x_8_2 = resnet_ff(x_8_1,      embedding, parameters,subkey = subkey, local_num_shift = 2, cfg=cfg, param_asso= param_asso,sub_model_num=3)
        # maxpool (changes shape)
        x_4_0 = naive_downsample_2d(x_8_2, factor = maxpool_factor)

        # Down ResNet 4
        x_4_1 = resnet_ff(x_4_0,    embedding, parameters,subkey = subkey, local_num_shift = 0, cfg=cfg, param_asso=param_asso, sub_model_num=4)
        x_4_2 = resnet_ff(x_4_1,      embedding, parameters,subkey = subkey, local_num_shift = 2, cfg=cfg, param_asso= param_asso,sub_model_num=4)
        

        ### MIDDLE ###
        # ff 5
        x = resnet_ff(            x_in = x_4_2, 
                                embedding = embedding, 
                                parameters = parameters, 
                                subkey = subkey, 
                                cfg = cfg.model, 
                                param_asso = param_asso, 
                                sub_model_num = 5, 
                                local_num_shift = 0, ) # 4x4 -> 4x4
        # attn 6
        x = attention(            x_in = x, 
                                embedding = embedding, 
                                parameters = parameters, 
                                subkey = subkey, 
                                cfg = cfg.model, 
                                param_asso = param_asso, 
                                sub_model_num = 6, 
                                local_num_shift = 0, ) # 4x4 -> 4x4
        # ff 7
        x = resnet_ff(            x_in = x, 
                                embedding = embedding, 
                                parameters = parameters, 
                                subkey = subkey, 
                                cfg = cfg.model, 
                                param_asso = param_asso, 
                                sub_model_num = 7, 
                                local_num_shift = 0, ) # 4x4 -> 4x4   C_out = 512


        # Up ResNet 8
        x = resnet_ff(jnp.concatenate([x,x_4_2],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=8, local_num_shift = 0)
        x = resnet_ff(jnp.concatenate([x,x_4_1],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=8, local_num_shift = 2)
        x = resnet_ff(jnp.concatenate([x,x_4_0],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=8, local_num_shift = 4)
        # Upsample
        x = upsample2d(           x, factor=upsampling_factor) # 16x16 -> 32x32

        # Up ResNet 9
        x = resnet_ff(jnp.concatenate([x,x_8_2],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=9, local_num_shift = 0)
        x = resnet_ff(jnp.concatenate([x,x_8_1],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=9, local_num_shift = 2)
        x = resnet_ff(jnp.concatenate([x,x_8_0],axis=-1), embedding, parameters,subkey=subkey, cfg=cfg, param_asso=param_asso, sub_model_num=9, local_num_shift = 4)
        # Upsample
        x = upsample2d(           x, factor=upsampling_factor) # 16x16 -> 32x32
        # print(x)


        # Up ResNet Attn 10
        x = resnet_ff_10_0(jnp.concatenate((x,x_16_2),axis=-1), embedding, parameters,subkey=subkey)
        x = attn_10_0(x, embedding, parameters, subkey = subkey[1], )
        x = resnet_ff_10_1(jnp.concatenate((x,x_16_1),axis=-1), embedding, parameters,subkey=subkey)
        x = attn_10_1(x, embedding, parameters, subkey = subkey[3], )
        x = resnet_ff_10_2(jnp.concatenate((x,x_16_0),axis=-1), embedding, parameters,subkey=subkey)
        x = attn_10_2(x, embedding, parameters, subkey = subkey[5], )
        # Upsample
        x = upsample2d(x, factor=upsampling_factor) # 16x16 -> 32x32


        # Up ResNet 11
        x = resnet_ff_11_0(jnp.concatenate([x,x_32_2],axis=-1), embedding, parameters, subkey=subkey)
        x = resnet_ff_11_1(jnp.concatenate([x,x_32_1],axis=-1), embedding, parameters, subkey=subkey)
        x = resnet_ff_11_2(jnp.concatenate([x,x_32_0],axis=-1), embedding, parameters, subkey=subkey)


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

