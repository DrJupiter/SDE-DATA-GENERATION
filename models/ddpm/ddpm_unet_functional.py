# stop prelocation of memory


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

def naive_downsample_2d(x, factor=2):
  _N, H, W, C = x.shape
  x = jnp.reshape(x, [-1, H // factor, factor, W // factor, factor, C])
  return jnp.mean(x, axis=[2, 4])

def upsample2d(x, factor=2):
    """
    Upsamling function\n
    Dublicates values to increase shape\n
    credit to SDE paper: https://github.com/yang-song/score_sde/blob/main/models/up_or_down_sampling.py
    """ 
    # get shapes of input
    B, H, W, C = x.shape
    # rehape it to BxHx1xWx1xC
    x = jnp.reshape(x, [-1, H, 1, W, 1, C])
    
    # dubilicate values factor number of times
    x = jnp.tile(x, [1, 1, factor, 1, factor, 1])

    # Collect the shapes back into the desized "shape", resulting in and increase in H and W, by the factors magnitude.
    return jnp.reshape(x, [-1, H * factor, W * factor, C])

def get_timestep_embedding(timesteps, embedding_dim: int):
        """
        For math behind this see paper.\n
        timesteps: array of ints describing the timestep each "picture" of the batch is perturbed to.\n
        timesteps.shape = B\n
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        \n
        Credit to DDPM (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90)
        \n I just converted it to jax.
        """
        assert len(timesteps.shape) == 1 # and timesteps.dtype == tf.int32

        half_dim = embedding_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.int32) * -emb)
        emb = jnp.int32(timesteps)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        if embedding_dim % 2 == 1:  # zero pad if uneven number
            emb = jnp.pad(emb, [[0, 0], [0, 1]])
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb

def resnet(x_in, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift):
    # store local parameter "location" for use in forward 
    sub_model_num = sub_model_num
    local_num_shift = local_num_shift

    # get shapes of all convolution channels, so we can extract the correct ones given our local parameter location
    # conv_shapes = cfg.parameters.conv_channels

    # Find which and amount parameters this model needs
    prior = param_asso[:sub_model_num].sum(axis=0)
    current = param_asso[:sub_model_num+1].sum(axis=0)

    # get the indexes for each type of parameter we need
    conv_params_idx = range(prior[0],current[0])
    time_lin_params_idx = range(prior[2],current[2])

    # Get linear parameters needed later
    w = parameters[2][0][time_lin_params_idx[local_num_shift//2]]
    b = parameters[2][1][time_lin_params_idx[local_num_shift//2]]

    # Apply the function to the input data
    x = x_in # vmap(batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1) # batchnorm
    x = nn.relu(x)
    x = lax.conv_general_dilated( 
            lhs = x,    
            rhs = parameters[0][conv_params_idx[0+local_num_shift]], 
            window_strides = [1,1], 
            padding = 'same',
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
            )
    x = nn.relu(x)
    x = x + (jnp.matmul(nn.relu(embedding), w)+b)[:, None, None, :] # introducing time embedding
    x# = vmap(batchnorm1,axis_name="batch")(x.transpose(0,3,2,1)).transpose(0,3,2,1) 
    x = nn.relu(x)
    #x = dropout(x,key = subkey)
    x = lax.conv_general_dilated( 
            lhs = x,    
            rhs = parameters[0][conv_params_idx[1+local_num_shift]], 
            window_strides = [1,1], 
            padding = 'same',
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
            )
    return x

def resnet_ff(x_in, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift):
    # store local parameter "location" for use in forward 
    sub_model_num = sub_model_num
    local_num_shift = local_num_shift
    
    # Find which and amount parameters this model needs
    prior = param_asso[:sub_model_num].sum(axis=0)
    current = param_asso[:sub_model_num+1].sum(axis=0)

    # get the indexes for each type of parameter we need. 1 corresponds to skip connections
    s_lin_params = range(prior[1],current[1])

    w = parameters[1][0][s_lin_params[0+local_num_shift//2]]
    b = parameters[1][1][s_lin_params[0+local_num_shift//2]]

    # call chain, built on resnet
    x = resnet(x_in, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift)

    # perform skip connection
    x_in = jnp.einsum('bhwc,cC->bhwC', x, w) + b

    # add and return
    return x+x_in

def attention(x_in, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift):
    # store atten shapes for later use
    # attn_shapes = cfg.parameters.attention_linear

    # Find which and amount parameters this model needs
    prior = param_asso[:sub_model_num].sum(axis=0)
    current = param_asso[:sub_model_num+1].sum(axis=0)

    # get the indexes for each type of parameter we need. 3 corresponds to attention
    attn_lin_params_idx = range(prior[3],current[3])

    # Get shape for reshapes later
    B, H, W, C = x_in.shape

    # Get parameters needed for call
    w1 = parameters[3][0][attn_lin_params_idx[0+local_num_shift]]
    w2 = parameters[3][0][attn_lin_params_idx[1+local_num_shift]]
    w3 = parameters[3][0][attn_lin_params_idx[2+local_num_shift]]
    w4 = parameters[3][0][attn_lin_params_idx[3+local_num_shift]]

    b1 = parameters[3][1][attn_lin_params_idx[0+local_num_shift]]
    b2 = parameters[3][1][attn_lin_params_idx[1+local_num_shift]]
    b3 = parameters[3][1][attn_lin_params_idx[2+local_num_shift]]
    b4 = parameters[3][1][attn_lin_params_idx[3+local_num_shift]]

    # normalization
    x = x_in #vmap(batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1)

    # qkv linear passes
    q = jnp.einsum('bhwc,cC->bhwC', x, w1)+b1
    k = jnp.einsum('bhwc,cC->bhwC', x, w2)+b2
    v = jnp.einsum('bhwc,cC->bhwC', x, w3)+b3

    # scaled dot production attention (sdpa)
    sdpa = jnp.einsum('bhwc,bHWc->bhwHW', q, k) / (jnp.sqrt(C))
    sdpa = sdpa.reshape(B, H, W, H * W)
    sdpa = nn.softmax(sdpa, -1)
    sdpa = sdpa.reshape(B, H, W, H, W)

    # compute the final outputs
    x = jnp.einsum('bhwHW,bHWc->bhwc', sdpa, v)
    x = jnp.einsum('bhwc,cC->bhwC', x, w4)+b4

    # sum and return
    return x+x_in

######################## Advanced building blocks ########################


def down_resnet(x_in, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift, maxpool_factor):
    # split keys to preserave randomness
    subkey = random.split(subkey*sub_model_num,2)

    # pass thruogh resnets
    x0 = resnet_ff(x_in,    embedding, parameters,subkey = subkey[0], local_num_shift = local_num_shift+0, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)
    x1 = resnet_ff(x0,      embedding, parameters,subkey = subkey[1], local_num_shift = local_num_shift+2, cfg=cfg, param_asso= param_asso,sub_model_num=sub_model_num)

    # maxpool (changes shape)
    x2 = naive_downsample_2d(x1, factor = maxpool_factor)

    return x0,x1,x2

def down_resnet_attn(x_in, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift, maxpool_factor):
    # split randomness key
    subkey = random.split(subkey*sub_model_num,2)

    # pass through resnet and attention in alternating manner
    x00 = resnet_ff(x_in,   embedding, parameters, subkey = subkey[0], local_num_shift = local_num_shift+0, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)
    x01 = attention(x00,    embedding, parameters, subkey = subkey[1], local_num_shift = local_num_shift+0, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)
    x10 = resnet_ff(x01,    embedding, parameters, subkey = subkey[2], local_num_shift = local_num_shift+2, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)
    x11 = attention(x10,    embedding, parameters, subkey = subkey[3], local_num_shift = local_num_shift+4, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)

    # maxpool (changes shapes)
    x2 = naive_downsample_2d(x11, factor = maxpool_factor)

    # returns outputs from attention and maxpool
    return x01,x11,x2

def up_resnet(x_in, x_res0, x_res1, x_res2, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift):
    # split randomness key
    subkey = random.split(subkey*sub_model_num,3)

    # pass through resnets with residual input concatenated to x:
    x = resnet_ff(jnp.concatenate((x_in,x_res0),axis=-1), embedding, parameters,subkey=subkey[0], cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num, local_num_shift = 0+local_num_shift)
    x = resnet_ff(jnp.concatenate((x,x_res1),axis=-1), embedding, parameters,subkey=subkey[1], cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num, local_num_shift = 2+local_num_shift)
    x = resnet_ff(jnp.concatenate((x,x_res2),axis=-1), embedding, parameters,subkey=subkey[2], cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num, local_num_shift = 4+local_num_shift)
    return x

def up_resnet_attn(x_in, x_res0, x_res1, x_res2, embedding, parameters, subkey, cfg, param_asso, sub_model_num, local_num_shift):
    # split randomness key
    subkey = random.split(subkey*sub_model_num,6)

    # perform forward pass of up_resnet with attention excluding the upsampling.
    x = resnet_ff(jnp.concatenate((x_in,x_res0),axis=-1), embedding, parameters,subkey=subkey[0], cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num, local_num_shift = 0+local_num_shift)
    x = attention(x, embedding, parameters, subkey = subkey[1], local_num_shift = local_num_shift+0, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)
    x = resnet_ff(jnp.concatenate((x,x_res1),axis=-1), embedding, parameters,subkey=subkey[2], cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num, local_num_shift = 2+local_num_shift)
    x = attention(x, embedding, parameters, subkey = subkey[3], local_num_shift = local_num_shift+4, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)
    x = resnet_ff(jnp.concatenate((x,x_res2),axis=-1), embedding, parameters,subkey=subkey[4], cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num, local_num_shift = 4+local_num_shift)
    x = attention(x, embedding, parameters, subkey = subkey[5], local_num_shift = local_num_shift+8, cfg=cfg, param_asso=param_asso, sub_model_num=sub_model_num)

    return x

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