
# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax

# Equinox

# TODO: register class with jax?
# Inherit from equinox module to solve this

# TODO: create smaller test model, maybe like down_resnet_attn -> up_resnet_attn, and see if it crashes 

# TODO: Try to remove all batchnorm, and see if it works (this can make it all into pure functions)

# TODO: Try with and without skip connections (just pass blank imgs in instead of correct imgs, so as to not have the loss propagate through those.)

# TODO: Worst case reimplement with equinox. Shouldnt take too long, as i got all the info, and have basically done it before.

######################## Basic building blocks ########################

from models.ddpm.building_blocks.ddpm_functions import down_resnet, down_resnet_attn, up_resnet, up_resnet_attn, upsample2d, naive_downsample_2d, get_timestep_embedding, resnet_ff, attention

######################## MODEL ########################
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
                                  maxpool_factor = 2) # 16x16 -> 8x8      C_out = 256

        d30,d31,d32 = down_resnet(x_in = d22, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[3], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 3, 
                                  local_num_shift = 0, 
                                  maxpool_factor = 2) # 8x8 -> 4x4        C_out = 512

        d40,d41,_   = down_resnet(x_in = d32, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[4], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 4, 
                                  local_num_shift = 0, 
                                  maxpool_factor = 1) # 4x4 -> 4x4        C_out = 512

        ## middle
        m = resnet_ff(            x_in = d41, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[5], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 5, 
                                  local_num_shift = 0, ) # 4x4 -> 4x4

        m = attention(            x_in = m, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[6], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 6, 
                                  local_num_shift = 0, ) # 4x4 -> 4x4

        m = resnet_ff(            x_in = m, 
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[7], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 7, 
                                  local_num_shift = 0, ) # 4x4 -> 4x4   C_out = 512

        ## up
        u = up_resnet(            x_in = m, 
                                  x_res0 = d41, 
                                  x_res1 = d40, 
                                  x_res2 = d32,
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[8], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 8, 
                                  local_num_shift = 0, ) # 4x4 -> 4x4   C_out = 512

        u = upsample2d(           u, factor=upsampling_factor) # 4x4 -> 8x8

        u = up_resnet(            x_in = u, 
                                  x_res0 = d31, 
                                  x_res1 = d30, 
                                  x_res2 = d22,
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[9], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 9, 
                                  local_num_shift = 0, ) # 8x8 -> 8x8   C_out = 512

        u = upsample2d(           u, factor=upsampling_factor) # 8x8 -> 16x16

        u = up_resnet_attn(       x_in = u, 
                                  x_res0 = d21, 
                                  x_res1 = d20, 
                                  x_res2 = d12,
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[10], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 10, 
                                  local_num_shift = 0, ) # 16x16 -> 16x16 C_out = 256

        u = upsample2d(           u, factor=upsampling_factor) # 16x16 -> 32x32

        u = up_resnet(            x_in = u, 
                                  x_res0 = d11, 
                                  x_res1 = d10, 
                                  x_res2 = d0,
                                  embedding = embedding, 
                                  parameters = parameters, 
                                  subkey = subkey[11], 
                                  cfg = cfg.model, 
                                  param_asso = param_asso, 
                                  sub_model_num = 11, 
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

if __name__ == "__name__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'