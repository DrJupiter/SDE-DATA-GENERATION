from jax import random
def get_parameters(cfg, key):
        key = random.PRNGKey(cfg.model.key)

        # TODO: REMOVE, THIS IS TEMP
        anti_blowup_factor = 0.07

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
            parameters[0].append(anti_blowup_factor*random.normal(subkey[i], ((kernel_size_h,kernel_size_w,in_channel,out_channel)), dtype=jnp.float32))
        
        # Liner and Bias parameters for Skip connections
        key, *subkey = random.split(key,len(skip_linear)+1)
        for i,(in_dims,out_dims) in enumerate(skip_linear): 
            parameters[1][0].append(anti_blowup_factor*random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[1][1].append(anti_blowup_factor*random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))

        # Liner and Bias parameters for time embedding (first the ones happening in ResNets)
        key, *subkey = random.split(key,len(time_embed_linear)+1)
        for i,(in_dims,out_dims) in enumerate(time_embed_linear): 
            parameters[2][0].append(anti_blowup_factor*random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[2][1].append(anti_blowup_factor*random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))

        # adding for the first layers of the embedding (Then for the ones initializing it)
        key, *subkey = random.split(key,len(embedding_parameters)+1)
        for i,(in_dims,out_dims) in enumerate(embedding_parameters): 
            parameters[2][0].append(anti_blowup_factor*random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[2][1].append(anti_blowup_factor*random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))

        # Liner and Bias parameters for Attention
        key, *subkey = random.split(key,len(attention_linear)+1)
        for i,(in_dims,out_dims) in enumerate(attention_linear): 
            parameters[3][0].append(anti_blowup_factor*random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
            parameters[3][1].append(anti_blowup_factor*random.normal(subkey[i], (1, out_dims), dtype=jnp.float32))
    
        # Loop over mpa and add the elements like a sum to copy, such that the initial and end values each model need to index for can be found
        # Maybe just pass this list into each and they find it for themselves during initialisation.
        # jnp.array(mpa)[:,0]

        return parameters, key