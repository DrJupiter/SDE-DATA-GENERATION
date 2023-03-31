# stop prelocation of memory
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# JAX
import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax


######################## Basic building blocks ########################
class resnet():
    def __init__(self, cfg, param_asso, sub_model_num, local_num_shift = 0) -> None:
        """Initialises the ResNet Block, see paper for further explanation of this class"""

        # store local parameter "location" for use in forward 
        self.sub_model_num = sub_model_num
        self.local_num_shift = local_num_shift

        # get shapes of all convolution channels, to initialize batchnorm correctly
        self.conv_shapes = cfg.parameters.conv_channels

        # Find which and amount parameters this model needs
        prior = param_asso[:sub_model_num].sum(axis=0)
        current = param_asso[:sub_model_num+1].sum(axis=0)

        # get the indexes for each type of parameter we need
        self.conv_params_idx = range(prior[0],current[0])
        self.time_lin_params_idx = range(prior[2],current[2])
        
        # initialize the batchnorm parameters we need for the forward call
        self.batchnorm0 = eqx.experimental.BatchNorm(
                input_size=self.conv_shapes[self.conv_params_idx[0+local_num_shift]][0],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                )
        self.batchnorm1 = eqx.experimental.BatchNorm(
                input_size=self.conv_shapes[self.conv_params_idx[1+local_num_shift]][0],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                )
        
        # initialize dropout layer
        self.dropout = eqx.nn.Dropout(cfg.parameters.dropout_p,inference=cfg.parameters.inference)


    def forward(self,x_in,embedding,parameters,subkey):

        # Get linear parameters needed later
        w = parameters[2][0][self.time_lin_params_idx[self.local_num_shift//2]]
        b = parameters[2][1][self.time_lin_params_idx[self.local_num_shift//2]]

        # Apply the function to the input data
        x = vmap(self.batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1) # batchnorm
        x = nn.relu(x)
        x = lax.conv_general_dilated( 
                lhs = x,    
                rhs = parameters[0][self.conv_params_idx[0+self.local_num_shift]], 
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )
        x = nn.relu(x)
        x = x + (jnp.matmul(nn.relu(embedding), w)+b)[:, None, None, :] # introducing time embedding
        x = vmap(self.batchnorm1,axis_name="batch")(x.transpose(0,3,2,1)).transpose(0,3,2,1) 
        x = nn.relu(x)
        x = self.dropout(x,key = subkey)
        x = lax.conv_general_dilated( 
                lhs = x,    
                rhs = parameters[0][self.conv_params_idx[1+self.local_num_shift]], 
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )
        return x

class resnet_ff():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0) -> None:
        """Initialises the ResNet Linear Block, see paper for further explanation of this class"""

        # store local parameter "location" for use in forward 
        self.sub_model_num = sub_model_num
        self.local_num_shift = local_num_shift
        
        # Find which and amount parameters this model needs
        prior = param_asso[:sub_model_num].sum(axis=0)
        current = param_asso[:sub_model_num+1].sum(axis=0)

        # get the indexes for each type of parameter we need. 1 corresponds to skip connections
        self.s_lin_params = range(prior[1],current[1])

        # initialize resnet class
        self.resnet = resnet(cfg, param_asso, sub_model_num, local_num_shift = self.local_num_shift)

    def forward(self, x_in, embedding, parameters,subkey):
        # get parameters for skip connection
        w = parameters[1][0][self.s_lin_params[0+self.local_num_shift//2]]
        b = parameters[1][1][self.s_lin_params[0+self.local_num_shift//2]]

        # call chain, built on resnet
        x = self.resnet.forward(x_in, embedding, parameters, subkey=subkey)

        # perform skip connection
        x_in = jnp.einsum('bhwc,cC->bhwC', x, w) + b

        # add and return
        return x+x_in

class attention():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0) -> None:

        # store atten shapes for later use
        self.attn_shapes = cfg.parameters.attention_linear

        # store local parameter "location" for use in forward 
        self.sub_model_num = sub_model_num
        self.local_num_shift = local_num_shift

        # Find which and amount parameters this model needs
        prior = param_asso[:sub_model_num].sum(axis=0)
        current = param_asso[:sub_model_num+1].sum(axis=0)

        # get the indexes for each type of parameter we need. 3 corresponds to attention
        self.attn_lin_params_idx = range(prior[3],current[3])

        # Initialize batchnorm
        self.batchnorm0 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=self.attn_shapes[self.attn_lin_params_idx[0+local_num_shift]][0],
                axis_name="batch",
                momentum=cfg.parameters.momentum,
                eps=cfg.parameters.eps,
                )

    def forward(self,x_in,parameters):

        # Get shape for reshapes later
        B, H, W, C = x_in.shape

        # Get parameters needed for call
        w1 = parameters[3][0][self.attn_lin_params_idx[0+self.local_num_shift]]
        w2 = parameters[3][0][self.attn_lin_params_idx[1+self.local_num_shift]]
        w3 = parameters[3][0][self.attn_lin_params_idx[2+self.local_num_shift]]
        w4 = parameters[3][0][self.attn_lin_params_idx[3+self.local_num_shift]]

        b1 = parameters[3][1][self.attn_lin_params_idx[0+self.local_num_shift]]
        b2 = parameters[3][1][self.attn_lin_params_idx[1+self.local_num_shift]]
        b3 = parameters[3][1][self.attn_lin_params_idx[2+self.local_num_shift]]
        b4 = parameters[3][1][self.attn_lin_params_idx[3+self.local_num_shift]]

        # normalization
        x = vmap(self.batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1)

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

class down_resnet():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0,maxpool_factor=2) -> None:
        # store local location variation for later use
        self.sub_model_num = sub_model_num

        # initialize submodels this model is built by
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = local_num_shift+0)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = local_num_shift+2)

        # initialize maxpool
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor)

    def forward(self, x_in, embedding, parameters, subkey):
        # split randomness key
        subkey = random.split(subkey*self.sub_model_num,2)

        # pass thruogh resnets
        x0 = self.resnet0.forward(x_in, embedding, parameters,subkey = subkey[0])
        x1 = self.resnet1.forward(x0, embedding, parameters,subkey = subkey[1])

        # maxpool (changes shape)
        x2 = vmap(self.maxpool2d,axis_name="batch")(x1.transpose(0,3,2,1)).transpose(0,3,2,1) 

        # return all three outputs
        return x0,x1,x2

class down_resnet_attn():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0,maxpool_factor=2) -> None:
        # store local location variation for later use
        self.sub_model_num = sub_model_num

        # initialize submodels this model is built by
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 2+local_num_shift)
        self.attn0 = attention(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.attn1 = attention(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor)

    def forward(self, x_in, embedding, parameters, subkey):

        # split randomness key
        subkey = random.split(subkey*self.sub_model_num,2)

        # pass through resnet and attention in alternating manner
        x00 = self.resnet0.forward(x_in, embedding, parameters,subkey=subkey[0])
        x01 = self.attn0.forward(x00,parameters)
        x10 = self.resnet1.forward(x01, embedding, parameters,subkey=subkey[1])
        x11 = self.attn1.forward(x10,parameters)
        
        # maxpool (changes shapes)
        x2 = vmap(self.maxpool2d,axis_name="batch")(x11.transpose(0,3,2,1)).transpose(0,3,2,1)

        # returns outputs from attention and maxpool
        return x01,x11,x2

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

class up_resnet():
    def __init__(self, cfg, param_asso, sub_model_num, local_num_shift = 0,upsampling_factor=2) -> None:
        # store local parameter "location" 
        self.sub_model_num = sub_model_num
        # store upsampling factor for forward call
        self.upsampling_factor = upsampling_factor

        # inititalize resnets
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 2+local_num_shift)
        self.resnet2 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)

    def forward(self, x, embedding, x_res0, x_res1, x_res2, parameters, subkey):
        
        # split randomness key
        subkey = random.split(subkey*self.sub_model_num,3)

        # pass through resnets with residual input concatenated to x:
        x = self.resnet0.forward(jnp.concatenate((x,x_res0),axis=-1), embedding, parameters,subkey=subkey[0])
        x = self.resnet1.forward(jnp.concatenate((x,x_res1),axis=-1), embedding, parameters,subkey=subkey[1])
        x = self.resnet2.forward(jnp.concatenate((x,x_res2),axis=-1), embedding, parameters,subkey=subkey[2])
        return x

class up_resnet_attn():
    def __init__(self, cfg, param_asso, sub_model_num, local_num_shift = 0, upsampling_factor=2) -> None:
        # store local parameter "location" 
        self.sub_model_num = sub_model_num
        # store upsampling factor for forward call
        self.upsampling_factor = upsampling_factor

        # initialize resnets and attention relative to location location, adding shifts to each
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 2+local_num_shift)
        self.resnet2 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)
        self.attn0 = attention(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.attn1 = attention(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)
        self.attn2 = attention(cfg, param_asso,sub_model_num, local_num_shift = 8+local_num_shift)

    def forward(self, x, embedding, x_res0, x_res1, x_res2, parameters, subkey):
        """Performs the up-resnet-attn as seen in the paper. Excluding performing the upscaling. This is done in ddpm."""

        # split randomness key
        subkey = random.split(subkey*self.sub_model_num,3)

        # perform forward pass of up_resnet with attention excluding the upsampling.
        x = self.resnet0.forward(jnp.concatenate((x,x_res0),axis=-1), embedding, parameters,subkey=subkey[0])
        x = self.attn0.forward(x,parameters)
        x = self.resnet1.forward(jnp.concatenate((x,x_res1),axis=-1), embedding, parameters,subkey=subkey[1])
        x = self.attn1.forward(x,parameters)
        x = self.resnet2.forward(jnp.concatenate((x,x_res2),axis=-1), embedding, parameters,subkey=subkey[2])
        x = self.attn2.forward(x,parameters)
        return x
