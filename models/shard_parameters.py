import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

def shard_timestep_embedding(params, sharding):

    ## 2x Linear
    # skip:
    params["w0"] = jax.device_put(params["w0"], sharding)
    params["b0"] = jax.device_put(params["b0"], sharding.reshape((1,len(jax.devices()))))

    # time:
    params["w1"] = jax.device_put(params["w1"], sharding)
    params["b1"] = jax.device_put(params["b1"], sharding.reshape((1,len(jax.devices()))))

    return params 

def shard_batchnorm(params, sharding):

    ## 1x Linear
    params["l"] = jax.device_put(params["l"], sharding)
    params["b"] = jax.device_put(params["b"], sharding.reshape((1,len(jax.devices()))))

    return params

def shard_conv(params, sharding):

    params = jax.device_put(params, sharding)

    return params

def shard_resnet_ff(params, sharding):

    ## 2x Linear
    # skip: 
    params["skip_w"] = jax.device_put(params["skip_w"], sharding)
    params["skip_b"] = jax.device_put(params["skip_b"], sharding.reshape((1,len(jax.devices()))))

    # time:
    params["time_w"] = jax.device_put(params["time_w"], sharding)
    params["time_b"] = jax.device_put(params["time_b"], sharding.reshape((1,len(jax.devices()))))

    # 2x Conv
    params["conv1_w"] = jax.device_put(params["conv1_w"], sharding.reshape((1,1,1,len(jax.devices()))))
    params["conv2_w"] = jax.device_put(params["conv2_w"], sharding.reshape((1,1,1,len(jax.devices()))))

    params["btchN1"] = shard_batchnorm(params["btchN1"], sharding)
    params["btchN2"] = shard_batchnorm(params["btchN2"], sharding)

    return params

def shard_attention(params, sharding):

    ## 4x Linear
    # q:
    params["q_w"] = jax.device_put(params["q_w"], sharding)
    params["q_b"] = jax.device_put(params["q_b"], sharding.reshape((1,len(jax.devices()))))

    # k:
    params["k_w"] = jax.device_put(params["k_w"], sharding)
    params["k_b"] = jax.device_put(params["k_b"], sharding.reshape((1,len(jax.devices()))))

    # v:
    params["v_w"] = jax.device_put(params["v_w"], sharding)
    params["v_b"] = jax.device_put(params["v_b"], sharding.reshape((1,len(jax.devices()))))

    # final
    params["f_w"] = jax.device_put(params["f_w"], sharding)
    params["f_b"] = jax.device_put(params["f_b"], sharding.reshape((1,len(jax.devices()))))

    params["btchN1"] = shard_batchnorm(params["btchN1"], sharding)

    return params

######################## Advanced building blocks ########################

def shard_down(params, sharding):

    params["r1"] = shard_resnet_ff(params["r1"], sharding)
    params["r2"] = shard_resnet_ff(params["r2"] , sharding)

    return params

def shard_down_attn(params, sharding):

    params["r1"] = shard_resnet_ff(params["r1"], sharding)
    params["r2"] = shard_resnet_ff(params["r2"], sharding)

    params["a1"] = shard_attention(params["a1"], sharding)
    params["a2"] = shard_attention(params["a2"], sharding)

    return params

def shard_up(params, sharding):

    params["r1"] = shard_resnet_ff(params, sharding)
    params["r2"] = shard_resnet_ff(params, sharding)
    params["r3"] = shard_resnet_ff(params, sharding)

    return params

def shard_up_attn(params, sharding):

    params["r1"] = shard_resnet_ff(params, sharding)
    params["r2"] = shard_resnet_ff(params, sharding)
    params["r3"] = shard_resnet_ff(params, sharding)

    params["a1"] = shard_attention(params, sharding)
    params["a2"] = shard_attention(params, sharding)
    params["a3"] = shard_attention(params, sharding)

    return params

###### FULL MODEL ######

def get_ddpm_unet(params, sharding):

    # if cfg.model.hyperparameters.sharding:
    n_devices = len(jax.devices())
    sharding = PositionalSharding(mesh_utils.create_device_mesh((n_devices,1)))

    # get time embedding func and params
    params["p_embed"] = shard_timestep_embedding(params, sharding)

    # get model funcs and params
    params["p_c1"] =       shard_conv(params, sharding)

    params["p_d1"] =       shard_down(params, sharding)
    params["p_da2"] = shard_down_attn(params, sharding)
    params["p_d3"] =       shard_down(params, sharding)
    params["p_d4"] =       shard_down(params, sharding)

    params["p_mr1"] = shard_resnet_ff(params, sharding)
    params["p_ma2"] = shard_attention(params, sharding)
    params["p_mr3"] = shard_resnet_ff(params, sharding)

    params["p_u1"] =         shard_up(params, sharding)
    params["p_u2"] =         shard_up(params, sharding)
    params["p_ua3"] =   shard_up_attn(params, sharding)
    params["p_u4"] =         shard_up(params, sharding)

    params["p_c2"] =       shard_conv(params, sharding)

    # define all the aprams in a dict
    # params = {"p_d1":p_d1, "p_da2":p_da2, "p_d3":p_d3, "p_d4":p_d4,  # down
    #           "p_u1":p_u1, "p_u2":p_u2, "p_ua3":p_ua3, "p_u4":p_u4,  # up
    #           "p_mr1":p_mr1, "p_ma2":p_ma2, "p_mr3":p_mr3, # middle
    #           "p_c1":p_c1, "p_c2":p_c2, # conv
    #           "p_embed": p_embed}  # time embedding

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

