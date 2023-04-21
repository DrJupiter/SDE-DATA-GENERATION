# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax import random
from jax import nn
from jax import lax
import jax

# Equinox
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

def get_timestep_embedding(cfg, key, embedding_dim: int, sharding):
        
    abf = cfg.model.hyperparameters.anti_blowup_factor
    time_dims = cfg.model.hyperparameters.time_embedding_dims
    subkey = random.split(key,4)

    ### Define parameters for the model
    params = {} # change to jnp array and use jax.numpy.append?

    ## 2x Linear
    # skip:
    params["w0"] = jax.device_put(abf*random.normal(subkey[0], (embedding_dim,time_dims), dtype=jnp.float32), sharding)
    params["b0"] = jax.device_put(abf*random.normal(subkey[1], (1, time_dims), dtype=jnp.float32), sharding)

    # time:
    params["w1"] = jax.device_put(abf*random.normal(subkey[2], (time_dims,time_dims), dtype=jnp.float32), sharding)
    params["b1"] = jax.device_put(abf*random.normal(subkey[3], (1, time_dims), dtype=jnp.float32), sharding)


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

        # apply linear layers:
        emb = jnp.einsum("hc,cC -> hC",emb,params["w0"])+ params["b0"]
        emb = nonlin(emb)
        emb = jnp.einsum("hc,cC -> hC",emb,params["w1"])+ params["b1"]

        return emb

    return apply_timestep_embedding, params 

######################## Basic building blocks ########################

def get_conv(cfg, key, in_C, out_C, sharding):
    kernel_size = cfg.model.hyperparameters.kernel_size
    abf = cfg.model.hyperparameters.anti_blowup_factor
    params = jax.device_put(abf*random.normal(key, ((kernel_size, kernel_size, in_C, out_C)), dtype=jnp.float32), sharding)

    return conv2d, params

def get_resnet_ff(cfg, key, in_C, out_C, sharding):
    
    abf = cfg.model.hyperparameters.anti_blowup_factor
    kernel_size = cfg.model.hyperparameters.kernel_size
    time_dims = cfg.model.hyperparameters.time_embedding_dims
    subkey = random.split(key,6)
    
    ### Define parameters for the model
    params = {} # change to jnp array and use jax.numpy.append?

    ## 2x Linear
    # skip: 
    params["skip_w"] = jax.device_put(abf*random.normal(subkey[0], (in_C,out_C), dtype=jnp.float32), sharding)
    params["skip_b"] = jax.device_put(abf*random.normal(subkey[1], (1, out_C), dtype=jnp.float32), sharding)

    # time:
    params["time_w"] = jax.device_put(abf*random.normal(subkey[2], (time_dims,out_C), dtype=jnp.float32), sharding)
    params["time_b"] = jax.device_put(abf*random.normal(subkey[3], (1, out_C), dtype=jnp.float32), sharding)

    # 2x Conv
    params["conv1_w"] = jax.device_put(abf*random.normal(subkey[4], ((kernel_size, kernel_size, in_C, out_C)), dtype=jnp.float32), sharding)
    params["conv2_w"] = jax.device_put(abf*random.normal(subkey[5], ((kernel_size, kernel_size, out_C, out_C)), dtype=jnp.float32), sharding)


    def resnet(x_in, embedding, params, subkey):

        ### Apply the function to the input data
        #x = x_in+0.0 # to counter memory share problems
        # vmap(batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1) # batchnorm
        x = nonlin(x_in)
        # print(x.shape)
        # print(params["conv1_w"].shape)
        # print(conv2d(x[0],params["conv1_w"]))
        # print(jnp.sum(pmap(lambda x,w: conv2d(x,w), in_axes = (0,None))(x,params["conv1_w"])))
        x = conv2d(x, params["conv1_w"])
        x = nonlin(x)
        # x = x + linear(nonlin(embedding), params["time_w"], params["time_b"])[:, None, None, :] # introducing time embedding
        x = x + (jnp.einsum('wc,cC->wC',nonlin(embedding),params["time_w"])+params["time_b"])[:,None,None,:]
        # x = p_add(x, p_add(p_linear(p_nonlin(embedding), params["time_w"]), params["time_b"])[:, None, None, :]) # introducing time embedding
        #x = vmap(batchnorm1,axis_name="batch")(x.transpose(0,3,2,1)).transpose(0,3,2,1) 
        x = nonlin(x)
        #x = dropout(x,key = subkey)
        x = conv2d(x, w = params["conv2_w"])

        # perform skip connection
        x_skip = linear(x_in, params["skip_w"],params["skip_b"])

        # add skip connections
        return x + x_skip

    return resnet, params

def get_attention(cfg, key, in_C, out_C, sharding):
    assert in_C == out_C, "in and out channels should be identical"

    abf = cfg.model.hyperparameters.anti_blowup_factor
    subkey = random.split(key,8)
    
    ### Define parameters for the model
    params = {} # change to jnp array and use jax.numpy.append?

    ## 4x Linear
    # q:
    params["q_w"] = jax.device_put(abf*random.normal(subkey[0], (in_C,out_C), dtype=jnp.float32), sharding)
    params["q_b"] = jax.device_put(abf*random.normal(subkey[1], (1, out_C), dtype=jnp.float32), sharding)

    # k:
    params["k_w"] = jax.device_put(abf*random.normal(subkey[2], (in_C,out_C), dtype=jnp.float32), sharding)
    params["k_b"] = jax.device_put(abf*random.normal(subkey[3], (1, out_C), dtype=jnp.float32), sharding)

    # v:
    params["v_w"] = jax.device_put(abf*random.normal(subkey[4], (in_C,out_C), dtype=jnp.float32), sharding)
    params["v_b"] = jax.device_put(abf*random.normal(subkey[5], (1, out_C), dtype=jnp.float32), sharding)

    # final
    params["f_w"] = jax.device_put(abf*random.normal(subkey[6], (out_C,out_C), dtype=jnp.float32), sharding)
    params["f_b"] = jax.device_put(abf*random.normal(subkey[7], (1, out_C), dtype=jnp.float32), sharding)


    def attn(x_in, embedding, params, subkey):

        # Get shape for reshapes later
        B, H, W, C = x_in.shape

        # normalization
        #vmap(batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1)

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
        return x+ x_in

    return attn, params

######################## Advanced building blocks ########################

def get_down(cfg, key, in_C, out_C, sharding):

    resnet1, params1 = get_resnet_ff(cfg, key, in_C, out_C, sharding)
    resnet2, params2 = get_resnet_ff(cfg, key, out_C, out_C, sharding)

    params = {"r1":params1, "r2": params2}

    def down(x_in, embedding, params, subkey, factor):
        x0 = resnet1(x_in, embedding, params["r1"], subkey)
        x1 = resnet2(x0, embedding, params["r2"], subkey)
        x2 = naive_downsample_2d(x1, factor=factor)

        return x0,x1,x2

    return down, params

def get_down_attn(cfg, key, in_C, out_C, sharding):

    resnet1, params1 = get_resnet_ff(cfg, key, in_C, out_C, sharding)
    attn1, params_a1 = get_attention(cfg, key, out_C, out_C, sharding)
    resnet2, params2 = get_resnet_ff(cfg, key, out_C, out_C, sharding)
    attn2, params_a2 = get_attention(cfg, key, out_C, out_C, sharding)

    params = {"r1":params1, "r2": params2,"a1": params_a1,"a2": params_a2}

    def down_attn(x_in, embedding, params, subkey, factor):
        x0 = resnet1(x_in, embedding, params["r1"], subkey)
        x0a = attn1(x0, embedding, params["a1"], subkey)
        x1 = resnet2(x0a, embedding, params["r2"], subkey)
        x1a = attn2(x1, embedding, params["a2"], subkey)
        x2 = naive_downsample_2d(x1a, factor=factor)

        return x0a,x1a,x2

    return down_attn, params

def get_up(cfg, key, in_C, out_C, residual_C: list, sharding):

    resnet1, params1 = get_resnet_ff(cfg, key, int(in_C+residual_C[0]), out_C, sharding)
    resnet2, params2 = get_resnet_ff(cfg, key, int(out_C+residual_C[1]), out_C, sharding)
    resnet3, params3 = get_resnet_ff(cfg, key, int(out_C+residual_C[2]), out_C, sharding)

    params = {"r1":params1, "r2": params2,"r3": params3}

    def up(x, x_res1, x_res2, x_res3, embedding, params, subkey, factor):
        x = resnet1(jnp.concatenate([x,x_res1],-1), embedding, params["r1"], subkey)
        x = resnet2(jnp.concatenate([x,x_res2],-1), embedding, params["r2"], subkey)
        x = resnet3(jnp.concatenate([x,x_res3],-1), embedding, params["r3"], subkey)
        x = upsample2d(x, factor=factor)
    
        return x

    return up, params

def get_up_attn(cfg, key, in_C, out_C, residual_C: list, sharding):

    resnet1, params1 = get_resnet_ff(cfg, key, int(in_C+residual_C[0]), out_C, sharding)
    resnet2, params2 = get_resnet_ff(cfg, key, int(out_C+residual_C[1]), out_C, sharding)
    resnet3, params3 = get_resnet_ff(cfg, key, int(out_C+residual_C[2]), out_C, sharding)
    attn1, params_a1 = get_attention(cfg, key, out_C, out_C, sharding)
    attn2, params_a2 = get_attention(cfg, key, out_C, out_C, sharding)
    attn3, params_a3 = get_attention(cfg, key, out_C, out_C, sharding)

    params = {"r1":params1, "r2": params2,"r3": params3,"a1": params_a1,"a2": params_a2,"a3": params_a3}

    def up_attn(x, x_res1, x_res2, x_res3, embedding, params, subkey, factor):
        x = resnet1(jnp.concatenate([x,x_res1],-1), embedding, params["r1"], subkey)
        x = attn1(x, embedding, params["a1"], subkey)
        x = resnet2(jnp.concatenate([x,x_res2],-1), embedding, params["r2"], subkey)
        x = attn2(x, embedding, params["a2"], subkey)
        x = resnet3(jnp.concatenate([x,x_res3],-1), embedding, params["r3"], subkey)
        x = attn3(x, embedding, params["a3"], subkey)
        x = upsample2d(x, factor=factor)

        return x

    return up_attn, params

if __name__ == "__main__":

    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'



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

    resnet, params = get_attention(cfg, jax.random.PRNGKey(22), 3, 3)

    out = resnet(jnp.ones((2,4,5,3),dtype=jnp.float32), jnp.ones((2,128),dtype=jnp.float32), params, jax.random.PRNGKey(2332))
    print(out.shape)