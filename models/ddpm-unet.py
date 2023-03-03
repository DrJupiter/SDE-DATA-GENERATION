# %%
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax.numpy as jnp
from jax import grad, jit, vmap 
from jax import random
from jax import nn
from jax import lax

import equinox as eqx
# %%

# Jax input denotions
# N - batch dimension
# H - spatial height
# W - spatial height
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension
# P - model Parameter count
 
# Conv2D - Done
# Maxpool - Done
# batchnorm - Done
# Dropout ? - 
# Relu - nn.relu(x)

class resnet():
    def __init__(self,cfg,sub_model_num) -> None:
        self.sub_model_num = sub_model_num
        self.batchnorm0 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num+0][1],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                )
        self.batchnorm1 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num+1][1],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.dropout = eqx.nn.dropout(cfg.parameters.dropout_p[sub_model_num])
        self.strides = cfg.parameters.kernel_stride[2*self.sub_model_num:2*self.sub_model_num+2]


    def forward(self,x_in,parameters,subkey=None):
        x = vmap(self.batchnorm0,axis_name="batch")(x_in) 
        x = nn.relu(x)
        x = lax.conv(
                lhs = x,    
                rhs = parameters[2*self.sub_model_num+0], 
                window_strides = self.strides[0],  
                padding = 'same'
                )
        x = nn.relu(x)
        x = x + nn.relu(jnp.matmul(x,parameters["dense"])+parameters["bias"])# residual FF
        x = vmap(self.batchnorm1,axis_name="batch")(x) 
        x = nn.relu(x)
        x = self.dropout(x,key = subkey)
        x = lax.conv(
                lhs = x,    
                rhs = parameters[2*self.sub_model_num+1], 
                window_strides = self.strides[1],  
                padding = 'same'
                )

        return x

class resnet_conv():
    def __init__(self,cfg,sub_model_num) -> None:
        self.sub_model_num = sub_model_num
        self.resnet = resnet(cfg,sub_model_num)
        self.stride = cfg.parameters.kernel_stride[sub_model_num]

    def forward(self,x_in,parameters,subkey=None):
        x = self.resnet.forward(x_in,parameters,subkey=None)
        
        # make change in input to make it downscalable (i assume)
        x_in = lax.conv(
                lhs = x_in,    
                rhs = parameters[2*self.sub_model_num+1], 
                window_strides = self.strides[1],  
                padding = 'same'
                )
        return x+x_in

class resnet_ff():
    def __init__(self,cfg,sub_model_num) -> None:
        self.sub_model_num = sub_model_num
        self.resnet = resnet(cfg,sub_model_num)
        self.stride = cfg.parameters.kernel_stride[sub_model_num]

    def forward(self,x_in,parameters,subkey=None):
        x = self.resnet.forward(x_in,parameters,subkey=None)
        w = None
        b = None
        # make change in input to make it downscalable (i assume)
        x_in = jnp.einsum('bhwc,cC->bhwC', x, w)+b
        return x+x_in

class attention():
    def __init__(self,cfg,sub_model_num) -> None:
        self.sub_model_num = sub_model_num
        self.batchnorm0 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num+0][1],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                )

    def forward(self,x_in,parameters):
        # Get shape for reshapes later
        B, H, W, C = x_in.shape

        # will be replaces with parameters
        w1 = jnp.arange(C**2).reshape(C,C)
        w2 = jnp.arange(C**2).reshape(C,C)
        w3 = jnp.arange(C**2).reshape(C,C)
        w4 = jnp.arange(C**2).reshape(C,C)

        b1 = jnp.arange(C)
        b2 = jnp.arange(C)
        b3 = jnp.arange(C)
        b4 = jnp.arange(C)

        # normalization
        x = vmap(self.batchnorm0,axis_name="batch")(x_in)

        # qkv linear passes
        q = jnp.einsum('bhwc,cC->bhwC', x, w1)+b1
        k = jnp.einsum('bhwc,cC->bhwC', x, w2)+b2
        v = jnp.einsum('bhwc,cC->bhwC', x, w3)+b3

        # scaled dot production attention (sdpa)
        sdpa = jnp.einsum('bhwc,bHWc->bhwHW', q, k) / (jnp.sqrt(C))
        sdpa = sdpa.reshape(B, H, W, H * W)
        sdpa = nn.softmax(sdpa, -1)
        sdpa = sdpa.reshape(B, H, W, H, W)

        x = jnp.einsum('bhwHW,bHWc->bhwc', sdpa, v)
        x = jnp.einsum('bhwc,cC->bhwC', x, w4)+b4

        return x+x_in

class down_resnet():
    def __init__(self,cfg,sub_model_num,maxpool_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.resnet = resnet_conv(cfg,sub_model_num)
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor)

    def forward(self, x_in, parameters):

        x = self.resnet.forward(x_in,parameters,subkey=None)
        x = vmap(self.maxpool2d,axis_name="batch")(x)
        return x

class down_resnet_attn():
    def __init__(self,cfg,sub_model_num,maxpool_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.resnet = resnet_conv(cfg,sub_model_num)
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor)
        self.attn = attention(cfg,sub_model_num)

    def forward(self, x_in, parameters):

        x = self.resnet.forward(x_in,parameters,subkey=None)
        x = self.attn.forward(x_in,parameters)
        x = vmap(self.maxpool2d,axis_name="batch")(x)
        return x

def upsample2d(x, factor=2):
    # stolen from https://github.com/yang-song/score_sde/blob/main/models/up_or_down_sampling.py
    _N, C, H, W = x.shape
    x = jnp.reshape(x, [-1, C, 1, H, 1, W])
    x = jnp.tile(x, [1, 1, factor, 1, factor, 1])
    return jnp.reshape(x, [-1, C, H * factor, W * factor])
# upsampled = upsample2d(input_x, factor=self.upsampling_factor)

class up_resnet():
    def __init__(self,cfg,sub_model_num,upsampling_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.upsampling_factor = upsampling_factor
        self.resnet = resnet_conv(cfg,sub_model_num)

    def forward(self, x_in, parameters):
        x = self.resnet.forward(x_in,parameters,subkey=None)
        x = upsample2d(x, factor=self.upsampling_factor)
        return x

class up_resnet_attn():
    def __init__(self,cfg,sub_model_num,upsampling_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.upsampling_factor = upsampling_factor
        self.resnet = resnet_conv(cfg,sub_model_num)
        self.attn = attention(cfg,sub_model_num)

    def forward(self, x_in, parameters):
        x = self.resnet.forward(x_in,parameters,subkey=None)
        x = self.attn.forward(x_in,parameters)
        x = upsample2d(x, factor=self.upsampling_factor)
        return x

class ddpm_unet():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        # down
        self.conv_resnet_0 = down_resnet(cfg,sub_model_num=0,maxpool_factor=2)
        self.conv_resnet_attn_1 = down_resnet_attn(cfg,sub_model_num=1,maxpool_factor=2)
        self.conv_resnet_2 = down_resnet(cfg,sub_model_num=2,maxpool_factor=2)
        self.conv_resnet_3 = down_resnet(cfg,sub_model_num=3,maxpool_factor=1) # no downsampling here

        # middle
        self.resnet4 = resnet_ff(cfg,sub_model_num=4)
        self.attn5 = attention(cfg,sub_model_num=5)
        self.resnet6 = resnet_ff(cfg,sub_model_num=6)

        # up
        self.conv_resnet_7 = down_resnet(cfg,sub_model_num=7,maxpool_factor=1) # no upsampling here
        self.conv_resnet_attn_8 = up_resnet_attn(cfg,sub_model_num=8,maxpool_factor=2)
        self.conv_resnet_9 = down_resnet(cfg,sub_model_num=9,maxpool_factor=2)
        self.conv_resnet_10 = down_resnet(cfg,sub_model_num=10,maxpool_factor=2)

        # end
        self.batchnorm_11 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[11][1], #takes in-channels as input_size
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                )
        self.stride = cfg.parameters.kernel_stride[12]


    def forward(self,x_in,parameters):
        # down
        d0 = self.conv_resnet_0.forward(x_in, parameters) # 32x32 -> 16x16
        d1 = self.conv_resnet_attn_1.forward(d0, parameters) # 16x16 -> 8x8
        d2 = self.conv_resnet_2.forward(d1, parameters) # 8x8 -> 4x4
        d3 = self.conv_resnet_3.forward(d2, parameters) # 4x4 -> 4x4 

        # middle
        m = self.resnet4.forward(d3,parameters,subkey=None) # 4x4 -> 4x4
        m = self.attn5.forward(m,parameters) # 4x4 -> 4x4
        m = self.resnet6.forward(m,parameters,subkey=None) # 4x4 -> 4x4

        # up
        u = self.conv_resnet_7.forward(jnp.concatenate((u,d3),axis=1),parameters) # 4x4 -> 4x4
        u = self.conv_resnet_attn_8.forward(jnp.concatenate((u,d2),axis=1),parameters) # 4x4 -> 8x8
        u = self.conv_resnet_9.forward(jnp.concatenate((u,d1),axis=1),parameters) # 8x8 -> 16x16
        u = self.conv_resnet_10.forward(jnp.concatenate((u,d0),axis=1),parameters) # 16x16 -> 32x32

        # end
        e = vmap(self.batchnorm_11,axis_name="batch")(u)
        e = nn.relu(e)
        e = lax.conv( 
                lhs = e,    # lhs = NCHW image tensor
                rhs = parameters[12], # rhs = OIHW conv kernel tensor
                window_strides = self.stride,  # window strides
                padding = 'same'
                )

    def get_parameters(self,cfg):
        key = random.PRNGKey(cfg.model.key)

        channels = cfg.model.parameters.channels
        kernel_sizes = cfg.model.parameters.kernel_sizes
        Linear_dims = cfg.model.parameters.Linear_dims

        key, *subkey = random.split(key,len(channels)+1)
        parameters = [[],[]]
        for i,((in_channel,out_channel),(kernel_size_h,kernel_size_w)) in enumerate(zip(channels,kernel_sizes)): 
            parameters[0].append(random.normal(subkey[i], ((out_channel,in_channel,kernel_size_h,kernel_size_w)), dtype=jnp.float32))
        
        key, *subkey = random.split(key,len(Linear_dims)+1)
        for i,(in_dims,out_dims) in enumerate(Linear_dims): 
            parameters[1].append(random.normal(subkey[i], (in_dims,out_dims), dtype=jnp.float32))
        
        return parameters
        
    def loss_mse(self, res, true):
        return jnp.mean((res-true)**2)

    def loss_fn(self, parameters, true_data):
        output = self.forward(true_data, parameters)
        loss = jnp.sum(output) #self.loss_mse(output, true_data)
        return loss


if __name__ == "__main__":
    # Andreas needs
    import sys
    sys.path.append("/media/sf_Bsc-Diffusion")
    # need ends

    from utils.utils import get_hydra_config
    cfg = get_hydra_config()
    # print(cfg.model)

    key = random.PRNGKey(69)

    img = jnp.zeros((
            cfg.model.parameters.batch_size,    # Batchsize
            cfg.model.parameters.channels[0][0],# channels
            cfg.model.parameters.img_h,         # h
            cfg.model.parameters.img_w),        # w
                dtype=jnp.float32)
    img = img.at[0, 0, 2:2+10, 2:2+10].set(1.0) 

    model = ddpm_unet(cfg.model)
    parameters = model.get_parameters(cfg)
    get_grad = grad(jit(model.loss_fn),0)
    get_loss = jit(model.loss_fn)
    # print("loss",get_loss(parameters, img))
    grads = get_grad(parameters, img)

    # print grads to make sure they work as intended
    for i,gradi in enumerate(grads):
        print(f"parameter{i} shape",gradi.shape)
        if jnp.sum(gradi) == 0:
            print(f"grad{i} is not being calculated")
#%%




#%%
class fisk():
    def sum_logistic(self,x,y):
        return jnp.sum(x**2*y)

fsk = fisk()
x_small = jnp.arange(3.)
y_small = jnp.arange(3.)
derivative_fn = grad(jit(fsk.sum_logistic),0)
derivative_fn2 = grad(jit(fsk.sum_logistic),1)
print(derivative_fn(x_small,y_small),derivative_fn2(x_small,y_small))

#%%
import string

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  print(einsum_str)
  return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)



# print(X.shape,We.shape)

# jnp.sum(jnp.einsum('bhwc,bHWc->bHWc', X, We))==jnp.sum(jnp.einsum('bhwc,bHWc->bhwc', X, We))
# jnp.sum(q) == jnp.sum(jnp.einsum('bhwc,bHWC->bhWC', X, We))


