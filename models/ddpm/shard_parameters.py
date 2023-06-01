import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# TODO: change sharding to named sharding, not essential but nice

def shard_timestep_embedding(params, linear_sharding, bias_sharding):

    ## 2x Linear
    # skip:
    params["w0"] = jax.device_put(params["w0"], linear_sharding)
    params["b0"] = jax.device_put(params["b0"], bias_sharding)

    # time:
    params["w1"] = jax.device_put(params["w1"], linear_sharding)
    params["b1"] = jax.device_put(params["b1"], bias_sharding)

    return params 

def shard_text_embedding(params, linear_sharding, bias_sharding):

    ## 2x Linear
    # skip:
    params["w0"] = jax.device_put(params["w0"], linear_sharding)
    params["b0"] = jax.device_put(params["b0"], bias_sharding)

    # time:
    params["w1"] = jax.device_put(params["w1"], linear_sharding)
    params["b1"] = jax.device_put(params["b1"], bias_sharding)

    return params 


def shard_text_data_embedding(params, linear_sharding, bias_sharding):

    ## 2x Linear
    # skip:
    params["w0"] = jax.device_put(params["w0"], linear_sharding)
    params["b0"] = jax.device_put(params["b0"], bias_sharding)

    return params 

def shard_batchnorm(params, linear_sharding, bias_sharding):

    ## 1x Linear
    params["l"] = jax.device_put(params["l"], linear_sharding)
    params["b"] = jax.device_put(params["b"], bias_sharding) 

    return params

def shard_conv(params, sharding):

    params = jax.device_put(params, sharding)

    return params

def shard_resnet_ff(params, linear_sharding, bias_sharding, convolution_sharding):

    ## 2x Linear
    # skip: 
    params["skip_w"] = jax.device_put(params["skip_w"], linear_sharding)
    params["skip_b"] = jax.device_put(params["skip_b"], bias_sharding)

    # time:
    params["time_w"] = jax.device_put(params["time_w"], linear_sharding)
    params["time_b"] = jax.device_put(params["time_b"], bias_sharding)

    # 2x Conv
    params["conv1_w"] = jax.device_put(params["conv1_w"], convolution_sharding)
    params["conv2_w"] = jax.device_put(params["conv2_w"], convolution_sharding)

    # params["btchN1"] = shard_batchnorm(params["btchN1"], sharding)
    # params["btchN2"] = shard_batchnorm(params["btchN2"], sharding)

    return params

def shard_attention(params, linear_sharding, bias_sharding):

    ## 4x Linear
    # q:
    params["q_w"] = jax.device_put(params["q_w"], linear_sharding)
    params["q_b"] = jax.device_put(params["q_b"], bias_sharding)

    # k:
    params["k_w"] = jax.device_put(params["k_w"], linear_sharding)
    params["k_b"] = jax.device_put(params["k_b"], bias_sharding)

    # v:
    params["v_w"] = jax.device_put(params["v_w"], linear_sharding)
    params["v_b"] = jax.device_put(params["v_b"], bias_sharding)

    # final
    params["f_w"] = jax.device_put(params["f_w"], linear_sharding)
    params["f_b"] = jax.device_put(params["f_b"], bias_sharding)

    return params

######################## Advanced building blocks ########################

def shard_down(params, linear_sharding, bias_sharding, convolution_sharding):

    params["r1"] = shard_resnet_ff(params["r1"], linear_sharding, bias_sharding, convolution_sharding)
    params["r2"] = shard_resnet_ff(params["r2"], linear_sharding, bias_sharding, convolution_sharding)

    return params

def shard_down_attn(params, linear_sharding, bias_sharding, convolution_sharding):

    params["r1"] = shard_resnet_ff(params["r1"], linear_sharding, bias_sharding, convolution_sharding)
    params["r2"] = shard_resnet_ff(params["r2"], linear_sharding, bias_sharding, convolution_sharding)

    params["a1"] = shard_attention(params["a1"], linear_sharding, bias_sharding)
    params["a2"] = shard_attention(params["a2"], linear_sharding, bias_sharding)

    return params

def shard_up(params, linear_sharding, bias_sharding, convolution_sharding):

    params["r1"] = shard_resnet_ff(params["r1"], linear_sharding, bias_sharding, convolution_sharding)
    params["r2"] = shard_resnet_ff(params["r2"], linear_sharding, bias_sharding, convolution_sharding)
    params["r3"] = shard_resnet_ff(params["r3"], linear_sharding, bias_sharding, convolution_sharding)

    return params

def shard_up_attn(params, linear_sharding, bias_sharding, convolution_sharding):

    params["r1"] = shard_resnet_ff(params["r1"], linear_sharding, bias_sharding, convolution_sharding)
    params["r2"] = shard_resnet_ff(params["r2"], linear_sharding, bias_sharding, convolution_sharding)
    params["r3"] = shard_resnet_ff(params["r3"], linear_sharding, bias_sharding, convolution_sharding)

    params["a1"] = shard_attention(params["a1"], linear_sharding, bias_sharding)
    params["a2"] = shard_attention(params["a2"], linear_sharding, bias_sharding)
    params["a3"] = shard_attention(params["a3"], linear_sharding, bias_sharding)

    return params

###### FULL MODEL ######

import utils.sharding
from jax.sharding import NamedSharding, PartitionSpec

def shard_ddpm_unet(cfg,params):




    # if cfg.model.hyperparameters.sharding:
    
    (primary, secondary), mesh = utils.sharding.get_sharding(cfg) 


    linear_sharding = NamedSharding(mesh,PartitionSpec(primary, secondary[0]))
    bias_sharding = NamedSharding(mesh,PartitionSpec(secondary[0],primary))

    convolution_sharding = NamedSharding(mesh, PartitionSpec(*secondary, primary))
    last_convolution_sharding = NamedSharding(mesh, PartitionSpec(*secondary[:-1], primary, secondary[-1]))


    # get time embedding func and params
    params["p_embed"] = shard_timestep_embedding(params["p_embed"], linear_sharding, bias_sharding)
    params["p_text_embed_data"] = shard_text_data_embedding(params["p_text_embed_data"], linear_sharding, bias_sharding)
    params["p_text_embed"] = shard_text_embedding(params["p_text_embed"], linear_sharding, bias_sharding)

    # get model funcs and params
    params["p_c1"] =       shard_conv(params["p_c1"],convolution_sharding)

    params["p_d1"] =       shard_down(params["p_d1"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_da2"] = shard_down_attn(params["p_da2"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_d3"] =       shard_down(params["p_d3"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_d4"] =       shard_down(params["p_d4"], linear_sharding, bias_sharding, convolution_sharding)

    params["p_mr1"] = shard_resnet_ff(params["p_mr1"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_ma2"] = shard_attention(params["p_ma2"], linear_sharding, bias_sharding)
    params["p_mr3"] = shard_resnet_ff(params["p_mr3"], linear_sharding, bias_sharding, convolution_sharding)

    params["p_u1"] =         shard_up(params["p_u1"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_u2"] =         shard_up(params["p_u2"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_ua3"] =   shard_up_attn(params["p_ua3"], linear_sharding, bias_sharding, convolution_sharding)
    params["p_u4"] =         shard_up(params["p_u4"], linear_sharding, bias_sharding, convolution_sharding)

    params["p_c2"] =       shard_conv(params["p_c2"],last_convolution_sharding) 

    # return func and parameters
    return params

if __name__ == "__main__":

    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

    param = {"l": jax.numpy.ones((10,10)), "b": jax.numpy.arange(10).reshape(1,10)}
    param2 = {"s": param}
    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,1)))

    jax.debug.visualize_array_sharding(param2["s"]["l"])

    param = shard_batchnorm(param2["s"], sharding)

    jax.debug.visualize_array_sharding(param2["s"]["l"])


