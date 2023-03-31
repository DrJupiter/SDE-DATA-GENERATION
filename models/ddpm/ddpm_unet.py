

# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax

# Equinox
import equinox as eqx

# TODO: register class with jax?
# Inherit from equinox module to solve this

# TODO: create smaller test model, maybe like down_resnet_attn -> up_resnet_attn, and see if it crashes 

# TODO: Try to remove all batchnorm, and see if it works (this can make it all into pure functions)

# TODO: Try with and without skip connections (just pass blank imgs in instead of correct imgs, so as to not have the loss propagate through those.)

# TODO: Worst case reimplement with equinox. Shouldnt take too long, as i got all the info, and have basically done it before.

######################## MODEL ########################

from models.ddpm.building_blocks.ddpm_class import down_resnet, down_resnet_attn, up_resnet, up_resnet_attn, upsample2d, resnet_ff, attention


class ddpm_unet():
    def __init__(self,cfg) -> None:

        # Figure out reshape shape: B x W x H x C
        self.data_shape = cfg.dataset.shape 

        # store for later use
        self.cfg = cfg.model
        conv_shapes = cfg.model.parameters.conv_channels
        self.upsampling_factor = self.cfg.parameters.upsampling_factor
        self.embedding_dims = self.cfg.parameters.time_embedding_dims


        # get parameter assignments, to pass into submodels
        param_asso = jnp.array(self.cfg.parameters.model_parameter_association)

        # initialize submodels
        ## down
        self.resnet_1 = down_resnet(cfg.model, param_asso,sub_model_num=1,maxpool_factor=2)
        self.resnet_attn_2 = down_resnet_attn(cfg.model, param_asso,sub_model_num=2,maxpool_factor=2)
        self.resnet_3 = down_resnet(cfg.model, param_asso,sub_model_num=3,maxpool_factor=2)
        self.resnet_4 = down_resnet(cfg.model, param_asso,sub_model_num=4,maxpool_factor=1) # no downsampling here

        ## middle
        self.resnet_5 = resnet_ff(cfg.model, param_asso,sub_model_num=5)
        self.attn_6 = attention(cfg.model, param_asso,sub_model_num=6)
        self.resnet_7 = resnet_ff(cfg.model, param_asso,sub_model_num=7)

        ## up
        self.resnet_8 = up_resnet(cfg.model, param_asso,sub_model_num=8)
        self.resnet_9 = up_resnet(cfg.model, param_asso,sub_model_num=9)
        self.resnet_attn_10 = up_resnet_attn(cfg.model, param_asso,sub_model_num=10)
        self.resnet_11 = up_resnet(cfg.model, param_asso,sub_model_num=11)

        ## end
        self.batchnorm_12 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=conv_shapes[-1][0], # Its input is equal to last conv input, as this doesnt change shape
                axis_name="batch",
                momentum=cfg.model.parameters.momentum,
                eps=cfg.model.parameters.eps,
                # channelwise_affine=True
                )

    def forward(self, x_in, timesteps, parameters, key):

        # Save initial shape, so we can transform the output pack into this shape
        in_shape = x_in.shape

        # Transform input into the image shape
        x_in = x_in.reshape(self.data_shape)

        # Split key to preserve randomness
        key, *subkey = random.split(key,13)

        # Get parameters for Timestep embedding
        em_w1 = parameters[2][0][-2]
        em_w2 = parameters[2][0][-1]
        em_b1 = parameters[2][1][-2]
        em_b2 = parameters[2][1][-1]

        # Create the embedding given the timesteps
        embedding = self.get_timestep_embedding(timesteps, embedding_dim = self.embedding_dims) # embedding -> dense -> nonlin -> dense (Shape = Bx512)
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
        d10,d11,d12 = self.resnet_1.forward(       d0, embedding, parameters, subkey = subkey[1]) # 32x32 -> 16x16     C_out = 128
        d20,d21,d22 = self.resnet_attn_2.forward(  d12, embedding, parameters, subkey = subkey[2]) # 16x16 -> 8x8      C_out = 256
        d30,d31,d32 = self.resnet_3.forward(       d22, embedding, parameters, subkey = subkey[3]) # 8x8 -> 4x4        C_out = 512
        d40,d41,_   = self.resnet_4.forward(       d32, embedding, parameters, subkey = subkey[4]) # 4x4 -> 4x4        C_out = 512

        ## middle
        m = self.resnet_5.forward(                 d41, embedding, parameters,subkey = subkey[5]) # 4x4 -> 4x4
        m = self.attn_6.forward(                   m, parameters) # 4x4 -> 4x4
        m = self.resnet_7.forward(                 m, embedding, parameters,subkey = subkey[7]) # 4x4 -> 4x4   C_out = 512

        ## up
        u = self.resnet_8.forward(          m, embedding, x_res0=d41, x_res1=d40, x_res2=d32, parameters=parameters, subkey = subkey[8]) # 4x4 -> 4x4   C_out = 512
        u = upsample2d(                     u, factor=self.upsampling_factor) # 4x4 -> 8x8
        u = self.resnet_9.forward(          u, embedding, x_res0=d31, x_res1=d30, x_res2=d22, parameters=parameters, subkey = subkey[9]) # 8x8 -> 8x8   C_out = 512
        u = upsample2d(                     u, factor=self.upsampling_factor) # 8x8 -> 16x16
        u = self.resnet_attn_10.forward(    u, embedding, x_res0=d21, x_res1=d20, x_res2=d12, parameters=parameters, subkey = subkey[10]) # 16x16 -> 16x16 C_out = 256
        u = upsample2d(                     u, factor=self.upsampling_factor) # 16x16 -> 32x32
        u = self.resnet_11.forward(         u, embedding, x_res0=d11, x_res1=d10, x_res2=d0, parameters=parameters, subkey = subkey[11]) # 32x32 -> 32x32 C_out = 128

        #3 end
        e = vmap(self.batchnorm_12,axis_name="batch")(u.transpose(0,3,2,1)).transpose(0,3,2,1)
        e = nn.relu(e)
        e = lax.conv_general_dilated( 
                lhs = e,    
                rhs = parameters[0][-1], # kernel is the conv [0] and the last entry of these [-1]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )
        
        # return to shape loss can take (input shape)
        x_out = e.reshape(in_shape) 

        return x_out

    def get_parameters(self, cfg, key):
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
        
    def loss_mse(self, output, true_data):
        """used for test in this file"""
        return jnp.sum(output)

    def loss_fn(self, parameters, true_data, timestep, key):
        """used for test in this file"""
        output = self.forward(true_data, timestep, parameters, key = key)
        loss = self.loss_mse(output, true_data)
        return loss

    def get_timestep_embedding(self, timesteps, embedding_dim: int):
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

#%%
if __name__ == "__main__":
    # stop prelocation of memory
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# Andreas needs
    import sys
    sys.path.append("/media/sf_Bsc-Diffusion")
    # need ends

    from utils.utils import get_hydra_config
    cfg = get_hydra_config()
    # print(cfg.model)

    key = random.PRNGKey(69)
    from data.dataload import dataload
    train, _ = dataload(cfg)
    iter_train = iter(train)
    data, label = next(iter_train) 
    img = jnp.ones_like(data)
    model = ddpm_unet(cfg)
    img = img.reshape(model.shape)

#    img = jnp.ones((
#            cfg.train_and_test.train.batch_size,    # Batchsize
#            cfg.model.parameters.img_h,         # h
#            cfg.model.parameters.img_w,         # w
#            cfg.model.parameters.conv_channels[0][0],# channels
#            ),dtype=jnp.float32)
    B, H, W, C = img.shape
    parameters, key = model.get_parameters(cfg,key)
    get_grad = jit(grad(jit(model.loss_fn),0))
    # get_loss = jit(model.loss_fn)
    # print("loss",get_loss(parameters, img))

    # check if img channels == first conv channel:
    assert img.shape[-1] == parameters[0][0].shape[-2], f"The first conv channel doesnt correspond to img channels. Go into ddpm_unet.yaml and change it to {img.shape[-1]}"

    grads = get_grad(parameters, img, timestep = jnp.zeros(B), key = key)

    #%%
    # check if all parameters have grads:
    def check_grads_beq_zero(grads):
        print("False means that the sum of the entire gradient is 0. Which it normally not should be")

        ## check conv layers
        for i,gradi in enumerate(grads[0]):
            if jnp.sum(gradi)==0:
                print("conv layer:",i,jnp.sum(gradi)!=0)

        ## check skip linear layers
        # linear
        for i,gradi in enumerate(grads[1][0]):
            if jnp.sum(gradi)==0:
                print("skip linear layer:",i,jnp.sum(gradi)!=0)

        # bias
        for i,gradi in enumerate(grads[1][1]):
            if jnp.sum(gradi)==0:
                print("skip bias layer:",i,jnp.sum(gradi)!=0)

        # time embedding
        # linear
        for i,gradi in enumerate(grads[2][0]):
            if jnp.sum(gradi)==0:
                print("time linear layer:",i,jnp.sum(gradi)!=0)

        # bias
        for i,gradi in enumerate(grads[2][1]):
            if jnp.sum(gradi)==0:
                print("time bias layer:",i,jnp.sum(gradi)!=0)

        # attention
        # linear
        for i,gradi in enumerate(grads[3][0]):
            if jnp.sum(gradi)==0:
                print("attention linear layer:",i,jnp.sum(gradi)!=0)

        # bias
        for i,gradi in enumerate(grads[3][1]):
            if jnp.sum(gradi)==0:
                print("attention bias layer:",i,jnp.sum(gradi)!=0)

        print("If no 'False' appeared then it all worked out")


    check_grads_beq_zero(grads)
