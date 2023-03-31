# stop prelocation of memory
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax

######################## MODEL ########################

from models.ddpm.building_blocks.ddpm_functions import down_resnet, down_resnet_attn, up_resnet, up_resnet_attn, upsample2d, naive_downsample_2d, get_timestep_embedding, resnet_ff, attention

def get_ddpm_unet(cfg):

    def ddpm_unet(x_in, timesteps, parameters, key):

        # Transform input into the image shape
        x_in_shape = x_in.shape
        x_in = x_in.reshape(cfg.dataset.shape)

        # conv_shapes = cfg.model.parameters.conv_channels
        upsampling_factor = cfg.model.parameters.upsampling_factor
        embedding_dims = cfg.model.parameters.time_embedding_dims

        param_asso = jnp.array(cfg.model.parameters.model_parameter_association)

        # Split key to preserve randomness
        key, *subkey = random.split(key,13)

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
        d0 = lax.conv_general_dilated( 
                lhs = x_in,    
                rhs = parameters[0][0], # kernel is the conv [0] and the first entry i this [0]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC') # input and parameter shapes
                )
        d10,d11,d12 = down_resnet(x_in = d0, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[1], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 1, 
                                  local_num_shift = 0, 
                                  maxpool_factor = 2) # 32x32 -> 16x16     C_out = 128

        d20,d21,d22 = down_resnet_attn(x_in = d12, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[2], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 2, 
                                  local_num_shift = 0, 
                                  maxpool_factor = 1) # 16x16 -> 8x8      C_out = 256

        
        u = up_resnet_attn(       x_in = d22, 
                                  x_res0 = jnp.ones_like(d21), 
                                  x_res1 = jnp.ones_like(d20), 
                                  x_res2 = jnp.ones_like(d12),
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[3], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 3, 
                                  local_num_shift = 0, ) # 16x16 -> 16x16 C_out = 256

        u = upsample2d(           u, factor=upsampling_factor) # 16x16 -> 32x32

        u = up_resnet(            x_in = u, 
                                  x_res0 = jnp.ones_like(d11), 
                                  x_res1 = jnp.ones_like(d10), 
                                  x_res2 = jnp.ones_like(d0),
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[4], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 4, 
                                  local_num_shift = 0, ) # 32x32 -> 32x32 C_out = 128

        #3 end
        e = u # vmap(batchnorm_12,axis_name="batch")(u.transpose(0,3,2,1)).transpose(0,3,2,1)
        e = nn.relu(e)
        e = lax.conv_general_dilated( 
                lhs = e,    
                rhs = parameters[0][-1], # kernel is the conv [0] and the last entry of these [-1]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )

        # return to shape loss can take (input shape)
        x_out = e.reshape(x_in_shape) 

        return x_out


    return ddpm_unet 


############ Helper Functions ############

def get_parameters(cfg, key):
        key = random.PRNGKey(cfg.model.key)

        # Get parameters from config
        conv_channels = cfg.model.parameters.conv_channels
        kernel_sizes = cfg.model.parameters.kernel_sizes
        skip_linear = cfg.model.parameters.skip_linear
        time_embed_linear = cfg.model.parameters.time_embed_linear
        attention_linear = cfg.model.parameters.attention_linear
        embedding_parameters = cfg.model.parameters.embedding_parameters

        # initialize list for paramteres
        parameters = [[], [[],[]], [[],[]], [[],[]]] 
        # List of  [Conv, [sL,sB], [eL,eB], [aL,aB]], # details which si what 
        # L = Linear, B = Bias
        # s = skip_linear, e = time_embedding_linear, a = attention_linear

        ### Below the parameters will be apended to the parameter list

        # Conv2d parameters 
        key, *subkey = random.split(key,len(conv_channels)+1)
        for i,((in_channel,out_channel),(kernel_size_h,kernel_size_w)) in enumerate(zip(conv_channels,kernel_sizes)): 
            # kernal shouold be of the shape HWIO, I = in, O = out
            parameters[0].append(random.normal(subkey[i], ((kernel_size_h,kernel_size_w,in_channel,out_channel)), dtype=jnp.float32))
        
        # Liner and Bias parameters for Skip connections
        key, *subkey = random.split(key,len(skip_linear)+1)
        for i,(in_dims,out_dims) in enumerate(skip_linear): 
            parameters[1][0].append(random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[1][1].append(random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))

        # Liner and Bias parameters for time embedding (first the ones happening in ResNets)
        key, *subkey = random.split(key,len(time_embed_linear)+1)
        for i,(in_dims,out_dims) in enumerate(time_embed_linear): 
            parameters[2][0].append(random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[2][1].append(random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))

        # adding for the first layers of the embedding (Then for the ones initializing it)
        key, *subkey = random.split(key,len(embedding_parameters)+1)
        for i,(in_dims,out_dims) in enumerate(embedding_parameters): 
            parameters[2][0].append(random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[2][1].append(random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))

        # Liner and Bias parameters for Attention
        key, *subkey = random.split(key,len(attention_linear)+1)
        for i,(in_dims,out_dims) in enumerate(attention_linear): 
            parameters[3][0].append(random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[3][1].append(random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))
    
        # Loop over mpa and add the elements like a sum to copy, such that the initial and end values each model need to index for can be found
        # Maybe just pass this list into each and they find it for themselves during initialisation.
        # jnp.array(mpa)[:,0]

        return parameters, key