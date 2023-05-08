# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax import random
from jax import nn
from jax import lax
import jax

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

######################## very Basic building blocks ########################
def conv2d(x,w):
    out = lax.conv_general_dilated( 
            lhs = x,    
            rhs = w, 
            window_strides = [1,1], 
            padding = 'same',
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
            )      
    return out

p_conv2d = pmap(lambda x,w: conv2d(x,w), in_axes = (0,None))
"""
Input: x (B1, B2, H, W, In_C), w (H, W, In_C, Out_C)\n
Returns: out (B1, B2, H, W, Out_C)
"""

p_matmul = pmap(lambda x,w: jnp.einsum('bhwc,cC->bhwC',x,w), in_axes = (0,None))
"""
Input: x (B1, B2, H, W, C), w (In_C, Out_C)\n
Returns: out (B1, B2, H, W, Out_C)
"""
def matmul(x,w):
    """
    Input: x (B1, B2, H, W, C), w (In_C, Out_C)\n
    Returns: out (B1, B2, H, W, Out_C)
    """
    return jnp.einsum('bhwc,cC->bhwC',x,w)


p_linear = pmap(lambda x,w,b: jnp.einsum('bhwc,cC->bhwC',x,w)+b, in_axes = (0, None, None))
"""
Input: x (B1, B2, H, W, C), w (In_C, Out_C), b (1,Out_C)\n
Returns: out (B1, B2, H, W, Out_C)
"""

def linear(x,w,b):
    """
    Input: x (B1, B2, H, W, C), w (In_C, Out_C), b (1,Out_C)\n
    Returns: out (B1, B2, H, W, Out_C)
    """
    return jnp.einsum('bhwc,cC->bhwC',x,w)+b


ad = pmap(lambda v1,v2: v1+v2, in_axes = (0,0))
def p_add(vv1,vv2):
    """
    Input: x1 (B1, B2, H, W, C), x2 (B1, B2, H, W, C)\n
    Returns: out (B1, B2, H, W, C)
    """
    assert vv1.shape == vv2.shape
    return ad(vv1,vv2)

p_nonlin = pmap(lambda v1: nn.relu(v1), in_axes = (0))
"""
Input: x (B1, B2, H, W, C)\n
Returns: out (B1, B2, H, W, C)
"""

def nonlin(x):
    return nn.relu(x) # TODO: change to depend on config

#### ATTENTION STUFF ####

p_bmatmul_scale = pmap(lambda q, k, C: jnp.einsum('bhwc,bHWc->bhwHW',q,k)/jnp.sqrt(C), in_axes = (0,0,None))
"""
Input: x1 (B1, B2, H, W, C), x2 (B1, B2, H, W, C)\n
Returns: out (B1, B2, H, W, H, W)
"""

p_bmatmul = pmap(lambda sdpa, v: jnp.einsum('bhwHW,bHWc->bhwc', sdpa, v), in_axes = (0,0))
"""
Input: x1 (B1, B2, H, W, H, W), x2 (B1, B2, H, W, C)\n
Returns: out (B1, B2, H, W, C)
"""

p_softmax = pmap(lambda v1: nn.softmax(v1), in_axes = (0))
"""
Input: x (B1, B2, H, W, C)\n
Returns: out (B1, B2, H, W, Out_C)
"""


######################## Helper functions ########################

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

def get_text_data_embedding(cfg, key):
    abf = cfg.model.hyperparameters.anti_blowup_factor
    embedding_dim = cfg.text_embedding.shape # 1024
    shape = jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0])
    image_size = abs(jnp.prod(shape))

    # For relu
    initializer = jax.nn.initializers.he_normal()

    params = {}

    subkey = random.split(key,4)
    params["w0"] = initializer(subkey[0], (embedding_dim, image_size), dtype=jnp.float32)
    params["b0"] = initializer(subkey[1], (1, image_size), dtype=jnp.float32)

    def apply_text_data_embedding(data,text_embedding, params):
        data = data + jnp.matmul(text_embedding, params["w0"]) + params["b0"]
        return data
    
    return apply_text_data_embedding, params

def get_text_embedding(cfg, key):

    abf = cfg.model.hyperparameters.anti_blowup_factor
    text_dims = cfg.model.hyperparameters.text_embedding_dims # 256
    embedding_dim = cfg.text_embedding.shape # 1024
    subkey = random.split(key,4)

    ### Define parameters for the model
    params = {} # change to jnp array and use jax.numpy.append?

    initializer = jax.nn.initializers.he_normal()
    ## 2x Linear
    # skip:
    params["w0"] = abf*initializer(subkey[0], (embedding_dim,text_dims), dtype=jnp.float32)
    params["b0"] = abf*initializer(subkey[1], (1,text_dims), dtype=jnp.float32)

    #params["w0"] = abf*random.normal(subkey[0], (embedding_dim,text_dims), dtype=jnp.float32)
    #params["b0"] = abf*random.normal(subkey[1], (1,text_dims), dtype=jnp.float32)

    # time:
    params["w1"] = abf*initializer(subkey[2], (text_dims,text_dims), dtype=jnp.float32)
    params["b1"] = abf*initializer(subkey[3], (1,text_dims), dtype=jnp.float32)
    #params["w1"] = abf*random.normal(subkey[2], (text_dims,text_dims), dtype=jnp.float32)
    #params["b1"] = abf*random.normal(subkey[3], (1,text_dims), dtype=jnp.float32)

    # B X 1024
    def apply_text_embedding(text_embedding, params):
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
        if len(text_embedding.shape) == 1:
            text_embedding = text_embedding.reshape(1,-1)
        # apply linear layers:
        text_embedding = jnp.einsum("hc,cC -> hC",text_embedding,params["w0"])+ params["b0"]
        text_embedding = nonlin(text_embedding)
        text_embedding = jnp.einsum("hc,cC -> hC",text_embedding,params["w1"])+ params["b1"]

        return text_embedding

    return apply_text_embedding, params

def get_timestep_embedding(cfg, key):

    abf = cfg.model.hyperparameters.anti_blowup_factor
    time_dims = cfg.model.hyperparameters.time_embedding_dims
    subkey = random.split(key,4)
    embedding_dim = cfg.model.hyperparameters.time_embedding_inner_dim
    
    ### Define parameters for the model
    params = {} # change to jnp array and use jax.numpy.append?

    
    initializer = jax.nn.initializers.he_normal()

    ## 2x Linear

    # skip:
    params["w0"] = abf*initializer(subkey[0], (embedding_dim,time_dims), dtype=jnp.float32)
    params["b0"] = abf*initializer(subkey[1], (1,time_dims), dtype=jnp.float32)

    #params["w0"] = abf*random.normal(subkey[0], (embedding_dim,time_dims), dtype=jnp.float32)
    #params["b0"] = abf*random.normal(subkey[1], (1,time_dims), dtype=jnp.float32)

    # time:
    params["w1"] = abf*initializer(subkey[2], (time_dims,time_dims), dtype=jnp.float32)
    params["b1"] = abf*initializer(subkey[3], (1,time_dims), dtype=jnp.float32)
    #params["w1"] = abf*random.normal(subkey[2], (time_dims,time_dims), dtype=jnp.float32)
    #params["b1"] = abf*random.normal(subkey[3], (1,time_dims), dtype=jnp.float32)

    def apply_timestep_embedding(timesteps, params):
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

        # apply linear layers: # created by us
        emb = jnp.einsum("hc,cC -> hC",emb,params["w0"])+ params["b0"]
        emb = nonlin(emb)
        emb = jnp.einsum("hc,cC -> hC",emb,params["w1"])+ params["b1"]

        return emb

    return apply_timestep_embedding, params 

######################## Basic building blocks ########################

def get_dropout(cfg, key, in_C, out_C):

    p = cfg.model.hyperparameters.dropout_p
    
    def dropout(x_in, embedding, params, subkey):

        dropout_m = random.bernoulli(subkey, p=1-p, shape = x_in.shape)+0.0

        return jnp.multiply(x_in, dropout_m)
    
    return dropout, "_"

def get_batchnorm(cfg, key, in_C, out_C):
    """
    The following paper, says its like this during training: https://arxiv.org/pdf/1502.03167.pdf\\
    But other sources say that gamma and beta are scalars, som im a little confused.
    """
    subkey = random.split(key,2)
    abf = cfg.model.hyperparameters.anti_blowup_factor
    avg_length = cfg.model.hyperparameters.avg_length

    params = {}

    ## 1x Linear
    # correction: 
    params["l"] = abf*random.normal(subkey[0], (in_C,out_C), dtype=jnp.float32)
    params["b"] = abf*random.normal(subkey[1], (1,out_C), dtype=jnp.float32)

    # shifting terms:
    #params["running_mu"] = jnp.zeros((1), dtype=jnp.float32)
    #params["running_var"] = jnp.zeros((1), dtype=jnp.float32)

    def batchnorm(x_in, embedding, params, subkey):

        mu = jnp.mean(x_in,0)
        var = jnp.var(x_in,0)

        #params["running_mu"] = params["running_mu"]*(avg_length/(avg_length+1))+mu/avg_length
        #params["running_var"] = params["running_var"]*(avg_length/(avg_length+1))+var/avg_length

        return linear((x_in-mu)/jnp.sqrt(var+1e-5),params["l"],params["b"])

    def inference_batchnorm(x_in, embedding, params, subkey):

        return linear((x_in-params["running_mu"])/jnp.sqrt(params["running_var"]+1e-5),params["l"],params["b"])

    # return alternate version if we are running inference. This does not affect JIT as this called it not jitted.
    if inference:
        return inference_batchnorm, params
    else:
        return batchnorm, params

def get_conv(cfg, key, in_C, out_C,first=False):
    kernel_size = cfg.model.hyperparameters.kernel_size
    abf = cfg.model.hyperparameters.anti_blowup_factor

    initilizer = jax.nn.initializers.orthogonal()

    params = abf * initilizer(key, ((kernel_size, kernel_size, in_C, out_C)), dtype=jnp.float32)
    #params = abf*random.normal(key, ((kernel_size, kernel_size, in_C, out_C)), dtype=jnp.float32)
    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,))).reshape(1,1,1,n_devices)

    if first:
        sharding = sharding.reshape(1,1,1,-1)
    else:
        sharding = sharding.reshape(1,1,-1,1)

    @jit
    def j_conv2d(x,w):
        out = conv2d(x,w)
        out = jax.lax.with_sharding_constraint(out, sharding)
        return out

    return j_conv2d, params, j_conv2d

def get_resnet_ff(cfg, key, in_C, out_C):
    
    abf = cfg.model.hyperparameters.anti_blowup_factor
    kernel_size = cfg.model.hyperparameters.kernel_size
    time_dims = cfg.model.hyperparameters.time_embedding_dims
    subkey = random.split(key,6)
    
    ### Define parameters for the model
    params = {}


    initializer_lin = jax.nn.initializers.he_normal()
    initilizer_conv = jax.nn.initializers.orthogonal()
    ## 2x Linear
    # skip: 
    params["skip_w"] = abf*initializer_lin(subkey[0], (in_C,out_C), dtype=jnp.float32)
    params["skip_b"] = abf*initializer_lin(subkey[1], (1,out_C), dtype=jnp.float32)
    #params["skip_w"] = abf*random.normal(subkey[0], (in_C,out_C), dtype=jnp.float32)
    #params["skip_b"] = abf*random.normal(subkey[1], (1,out_C), dtype=jnp.float32)

    # time:
    params["time_w"] = abf*initializer_lin(subkey[2], (time_dims,out_C), dtype=jnp.float32)
    params["time_b"] = abf*initializer_lin(subkey[3], (1,out_C), dtype=jnp.float32)

    #params["time_w"] = abf*random.normal(subkey[2], (time_dims,out_C), dtype=jnp.float32)
    #params["time_b"] = abf*random.normal(subkey[3], (1,out_C), dtype=jnp.float32)

    # 2x Conv
    params["conv1_w"] = abf*initilizer_conv(subkey[4], ((kernel_size, kernel_size, in_C, out_C)), dtype=jnp.float32)
    params["conv2_w"] = abf*initilizer_conv(subkey[5], ((kernel_size, kernel_size, out_C, out_C)), dtype=jnp.float32)
    #params["conv1_w"] = abf*random.normal(subkey[4], ((kernel_size, kernel_size, in_C, out_C)), dtype=jnp.float32)
    #params["conv2_w"] = abf*random.normal(subkey[5], ((kernel_size, kernel_size, out_C, out_C)), dtype=jnp.float32)

    # batchnorm, params["btchN1"] = get_batchnorm(cfg, key, in_C, in_C, inference)
    # batchnorm2, params["btchN2"] = get_batchnorm(cfg, key, out_C, out_C, inference)
    dropout, _ = get_dropout(cfg, key, in_C, out_C)

    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,))).reshape(1,1,1,n_devices)

    @jit
    def resnet(x_in, embedding, params, subkey):

        ### Apply the function to the input data
            # x = batchnorm(x_in, embedding, params["btchN1"], subkey)
        x = nonlin(x_in)
        x = conv2d(x, params["conv1_w"])
        x = jax.lax.with_sharding_constraint(x, sharding)

        x = nonlin(x)
        x = x + (jnp.einsum('wc,cC->wC',nonlin(embedding),params["time_w"])+params["time_b"])[:,None,None,:]

            # x = batchnorm2(x, embedding, params["btchN2"], subkey)
        x = nonlin(x)
        x = dropout(x, embedding, None, subkey)

        # Enforce sharding shape to avoid ret_check failure when returning
        x = jax.lax.with_sharding_constraint(x, sharding)
        x = conv2d(x, w = params["conv2_w"])

        # perform skip connection
        x_skip = linear(x_in, params["skip_w"],params["skip_b"])

        # add skip connections
        x = x  + x_skip

        # Enforce sharding shape to avoid ret_check failure when returning
        x = jax.lax.with_sharding_constraint(x, sharding)

        return x

    @jit
    def inf_resnet(x_in, embedding, params, subkey):

        ### Apply the function to the input data
            # x = batchnorm(x_in, embedding, params["btchN1"], subkey)
        x = nonlin(x_in)
        x = conv2d(x, params["conv1_w"])
        x = jax.lax.with_sharding_constraint(x, sharding)

        x = nonlin(x)
        x = x + (jnp.einsum('wc,cC->wC',nonlin(embedding),params["time_w"])+params["time_b"])[:,None,None,:]

            # x = batchnorm2(x, embedding, params["btchN2"], subkey)
        x = nonlin(x)
        # x = dropout(x, embedding, None, subkey)

        # Enforce sharding shape to avoid ret_check failure when returning
        x = jax.lax.with_sharding_constraint(x, sharding)
        x = conv2d(x, w = params["conv2_w"])

        # perform skip connection
        x_skip = linear(x_in, params["skip_w"],params["skip_b"])

        # add skip connections
        x = x  + x_skip

        # Enforce sharding shape to avoid ret_check failure when returning
        x = jax.lax.with_sharding_constraint(x, sharding)

        return x

    return resnet, params, inf_resnet

def get_attention(cfg, key, in_C, out_C):
    assert in_C == out_C, "in and out channels should be identical"

    abf = cfg.model.hyperparameters.anti_blowup_factor
    subkey = random.split(key,8)
    
    ### Define parameters for the model
    params = {}

    initializer = nn.initializers.xavier_normal()
    ## 4x Linear
    # q:
    params["q_w"] = abf*initializer(subkey[0], (in_C,out_C), dtype=jnp.float32)
    params["q_b"] = abf*initializer(subkey[1], (1,out_C), dtype=jnp.float32)
    #params["q_w"] = abf*random.normal(subkey[0], (in_C,out_C), dtype=jnp.float32)
    #params["q_b"] = abf*random.normal(subkey[1], (1,out_C), dtype=jnp.float32)

    # k:
    params["k_w"] = abf*initializer(subkey[2], (in_C,out_C), dtype=jnp.float32)
    params["k_b"] = abf*initializer(subkey[3], (1,out_C), dtype=jnp.float32)
    #params["k_w"] = abf*random.normal(subkey[2], (in_C,out_C), dtype=jnp.float32)
    #params["k_b"] = abf*random.normal(subkey[3], (1,out_C), dtype=jnp.float32)

    # v:
    params["v_w"] = abf*initializer(subkey[4], (in_C,out_C), dtype=jnp.float32)
    params["v_b"] = abf*initializer(subkey[5], (1,out_C), dtype=jnp.float32)
    #params["v_w"] = abf*random.normal(subkey[4], (in_C,out_C), dtype=jnp.float32)
    #params["v_b"] = abf*random.normal(subkey[5], (1,out_C), dtype=jnp.float32)

    # final
    params["f_w"] = abf*initializer(subkey[6], (out_C,out_C), dtype=jnp.float32)
    params["f_b"] = abf*initializer(subkey[7], (1,out_C), dtype=jnp.float32)
    #params["f_w"] = abf*random.normal(subkey[6], (out_C,out_C), dtype=jnp.float32)
    #params["f_b"] = abf*random.normal(subkey[7], (1,out_C), dtype=jnp.float32)

    # batchnorm, params["btchN1"] = get_batchnorm(cfg, key, in_C, out_C)

    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,))).reshape(1,1,1,n_devices)

    @jit
    def attn(x_in, embedding, params, subkey):

        # Get shape for reshapes later
        B, H, W, C = x_in.shape

        # normalization
        # x = batchnorm(x_in, embedding, params["btchN1"], subkey)

        # qkv linear passes
        q = linear(x_in, params["q_w"], params["q_b"])
        k = linear(x_in, params["k_w"], params["k_b"])
        v = linear(x_in, params["v_w"], params["v_b"])

        # scaled dot production attention (sdpa)
        sdpa = jnp.einsum('bhwc,bHWc->bhwHW', q, k) / (jnp.sqrt(C))
        sdpa = sdpa.reshape(B, H, W, H * W)
        sdpa = nn.softmax(sdpa, -1)
        sdpa = sdpa.reshape(B, H, W, H, W)

        # compute the final outputs
        x = jnp.einsum('bhwHW,bHWc->bhwc', sdpa, v)
        x = linear(x, params["f_w"], params["f_b"])

        # sum and return
        x = x + x_in
        x = jax.lax.with_sharding_constraint(x, sharding)

        return x

    return attn, params, attn

######################## Advanced building blocks ########################

from functools import partial

def get_down(cfg, key, in_C, out_C, factor):

    resnet1, params1, inf_resnet1 = get_resnet_ff(cfg, key, in_C, out_C)
    resnet2, params2, inf_resnet2 = get_resnet_ff(cfg, key, out_C, out_C)

    params = {"r1":params1, "r2": params2}

    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,))).reshape(1,1,1,n_devices)

    factor = factor

    @jit #@partial(jax.jit, static_argnames=['factor'])
    def down(x_in, embedding, params, subkey):
        x_in = jax.lax.with_sharding_constraint(x_in, sharding)
        x0 = resnet1(x_in, embedding, params["r1"], subkey)
        x1 = resnet2(x0, embedding, params["r2"], subkey)
        x2 = naive_downsample_2d(x1, factor=factor)

        return x0,x1,x2

    @jit #@partial(jax.jit, static_argnames=['factor'])
    def inf_down(x_in, embedding, params, subkey):
        x_in = jax.lax.with_sharding_constraint(x_in, sharding)
        x0 = inf_resnet1(x_in, embedding, params["r1"], subkey)
        x1 = inf_resnet2(x0, embedding, params["r2"], subkey)
        x2 = naive_downsample_2d(x1, factor=factor)

        return x0,x1,x2

    return down, params, inf_down

def get_down_attn(cfg, key, in_C, out_C, factor):

    resnet1, params1, inf_resnet1 = get_resnet_ff(cfg, key, in_C, out_C)
    attn1, params_a1, inf_attn1 = get_attention(cfg, key, out_C, out_C)
    resnet2, params2, inf_resnet2 = get_resnet_ff(cfg, key, out_C, out_C)
    attn2, params_a2, inf_attn2 = get_attention(cfg, key, out_C, out_C)

    params = {"r1":params1, "r2": params2,"a1": params_a1,"a2": params_a2}

    factor=factor

    # @partial(jax.jit, static_argnames=['factor'])
    def down_attn(x_in, embedding, params, subkey):
        x0 = resnet1(x_in, embedding, params["r1"], subkey)
        x0a = attn1(x0, embedding, params["a1"], subkey)
        x1 = resnet2(x0a, embedding, params["r2"], subkey)
        x1a = attn2(x1, embedding, params["a2"], subkey)
        x2 = naive_downsample_2d(x1a, factor=factor)

        return x0a,x1a,x2
    
    def inf_down_attn(x_in, embedding, params, subkey):
        x0 = inf_resnet1(x_in, embedding, params["r1"], subkey)
        x0a = inf_attn1(x0, embedding, params["a1"], subkey)
        x1 = inf_resnet2(x0a, embedding, params["r2"], subkey)
        x1a = inf_attn2(x1, embedding, params["a2"], subkey)
        x2 = naive_downsample_2d(x1a, factor=factor)

        return x0a,x1a,x2

    return down_attn, params, inf_down_attn

def get_up(cfg, key, in_C, out_C, residual_C: list, factor):

    resnet1, params1, inf_resnet1 = get_resnet_ff(cfg, key, int(in_C+residual_C[0]), out_C)
    resnet2, params2, inf_resnet2 = get_resnet_ff(cfg, key, int(out_C+residual_C[1]), out_C)
    resnet3, params3, inf_resnet3 = get_resnet_ff(cfg, key, int(out_C+residual_C[2]), out_C)

    params = {"r1":params1, "r2": params2,"r3": params3}

    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,))).reshape(1,1,1,n_devices)

    factor = factor

    @jit #@partial(jax.jit, static_argnames=['factor'])
    def up(x, x_res1, x_res2, x_res3, embedding, params, subkey):
        x = jax.lax.with_sharding_constraint(x, sharding)
        x_res1 = jax.lax.with_sharding_constraint(x_res1, sharding)
        x_res2 = jax.lax.with_sharding_constraint(x_res2, sharding)
        x_res3 = jax.lax.with_sharding_constraint(x_res3, sharding)

        x = resnet1(jnp.concatenate([x,x_res1],-1), embedding, params["r1"], subkey)
        x = resnet2(jnp.concatenate([x,x_res2],-1), embedding, params["r2"], subkey)
        x = resnet3(jnp.concatenate([x,x_res3],-1), embedding, params["r3"], subkey)
        x = upsample2d(x, factor=factor)
    
        return x
    
    @jit #@partial(jax.jit, static_argnames=['factor'])
    def inf_up(x, x_res1, x_res2, x_res3, embedding, params, subkey):
        x = jax.lax.with_sharding_constraint(x, sharding)
        x_res1 = jax.lax.with_sharding_constraint(x_res1, sharding)
        x_res2 = jax.lax.with_sharding_constraint(x_res2, sharding)
        x_res3 = jax.lax.with_sharding_constraint(x_res3, sharding)

        x = inf_resnet1(jnp.concatenate([x,x_res1],-1), embedding, params["r1"], subkey)
        x = inf_resnet2(jnp.concatenate([x,x_res2],-1), embedding, params["r2"], subkey)
        x = inf_resnet3(jnp.concatenate([x,x_res3],-1), embedding, params["r3"], subkey)
        x = upsample2d(x, factor=factor)
    
        return x
    
    return up, params, inf_up

def get_up_attn(cfg, key, in_C, out_C, residual_C: list, factor):

    resnet1, params1, inf_resnet1 = get_resnet_ff(cfg, key, int(in_C+residual_C[0]), out_C)
    resnet2, params2, inf_resnet2 = get_resnet_ff(cfg, key, int(out_C+residual_C[1]), out_C)
    resnet3, params3, inf_resnet3 = get_resnet_ff(cfg, key, int(out_C+residual_C[2]), out_C)
    attn1, params_a1, inf_attn1 = get_attention(cfg, key, out_C, out_C)
    attn2, params_a2, inf_attn2 = get_attention(cfg, key, out_C, out_C)
    attn3, params_a3, inf_attn3 = get_attention(cfg, key, out_C, out_C)

    params = {"r1":params1, "r2": params2,"r3": params3,"a1": params_a1,"a2": params_a2,"a3": params_a3}

    factor = factor

    def up_attn(x, x_res1, x_res2, x_res3, embedding, params, subkey):
        x = resnet1(jnp.concatenate([x,x_res1],-1), embedding, params["r1"], subkey)
        x = attn1(x, embedding, params["a1"], subkey)
        x = resnet2(jnp.concatenate([x,x_res2],-1), embedding, params["r2"], subkey)
        x = attn2(x, embedding, params["a2"], subkey)
        x = resnet3(jnp.concatenate([x,x_res3],-1), embedding, params["r3"], subkey)
        x = attn3(x, embedding, params["a3"], subkey)
        x = upsample2d(x, factor=factor)

        return x
    
    def inf_up_attn(x, x_res1, x_res2, x_res3, embedding, params, subkey):
        x = inf_resnet1(jnp.concatenate([x,x_res1],-1), embedding, params["r1"], subkey)
        x = inf_attn1(x, embedding, params["a1"], subkey)
        x = inf_resnet2(jnp.concatenate([x,x_res2],-1), embedding, params["r2"], subkey)
        x = inf_attn2(x, embedding, params["a2"], subkey)
        x = inf_resnet3(jnp.concatenate([x,x_res3],-1), embedding, params["r3"], subkey)
        x = inf_attn3(x, embedding, params["a3"], subkey)
        x = upsample2d(x, factor=factor)

        return x
    
    return up_attn, params, inf_up_attn

if __name__ == "__main__":

    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding
    sharding = PositionalSharding(mesh_utils.create_device_mesh(4,)).reshape(4,1)

    print(jax.devices())

    from hydra import compose, initialize
    import hydra

    def get_hydra_config(config_path='../../../configs', job_name='test', version_base='1.3', config='defaults.yaml', overrides=[], reload=True):
        """
        !!! This function is meant for TESTING PURPOSES, not to be used
        in production. !!!

        Load the hydra config manually.
        
        (The parameters are the same as loading a hydraconfig normally)
        """

        if reload:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(config_path=config_path, job_name=job_name, version_base=version_base) 
        cfg = compose(config, overrides=overrides)
        return cfg
    
    cfg = get_hydra_config()
    key = jax.random.PRNGKey(2332)

    xx = jax.device_put(jnp.ones((4,4,8,4),dtype=jnp.float32)*10,sharding.reshape(len(jax.devices()),1,1,1))
    ww = jax.device_put(jnp.ones((4,128),dtype=jnp.float32)*10,sharding.reshape(len(jax.devices()),1))

    resnet, paramsr = get_resnet_ff(cfg, key, 4, 4)
    attn, paramsa = get_attention(cfg, key, 4, 4)
    down, paramsd = get_down_attn(cfg, key, 4, 4)
    up, paramsu = get_up_attn(cfg, key, 4, 4, residual_C=[0,0,0])

    outr = resnet(xx, ww, paramsr, key)
    outa = attn(xx, ww, paramsa, key)
    print(outr.shape, outa.shape)

    outd = down(xx, ww, paramsd, key, factor=1)
    outu = up(xx,xx,xx,xx, ww, paramsu, key, factor=1)
    print(outd.shape, outu.shape)