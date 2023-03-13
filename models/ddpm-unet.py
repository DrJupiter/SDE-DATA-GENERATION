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

# conv_shapes = cfg.model.parameters.conv_channels

# prior = param_asso[:sub_model_num].sum(axis=0)
# current = param_asso[:sub_model_num+1].sum(axis=0)

# self.conv_params_idx = range(prior[0],current[0])
# self.s_lin_params_idx = range(prior[1],current[1])
# self.time_lin_params_idx = range(prior[2],current[2])
# self.attn_lin_params_idx = range(prior[3],current[3])

# parameters[0][self.conv_params_idx[0+self.local_num_shift]] # 
# parameters[1][0][self.s_lin_params_idx[0+self.local_num_shift]] # Linear ..[1][1].. for bias
# parameters[2][0][self.time_lin_params_idx[0+self.local_num_shift]]
# parameters[3][0][self.attn_lin_params_idx[0+self.local_num_shift]]

# conv_shapes[conv_params_idx[0]][1]


class resnet():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0) -> None:
        self.sub_model_num = sub_model_num
        self.local_num_shift = local_num_shift

        self.conv_shapes = cfg.parameters.conv_channels

        prior = param_asso[:sub_model_num].sum(axis=0)
        current = param_asso[:sub_model_num+1].sum(axis=0)

        self.conv_params_idx = range(prior[0],current[0])
        self.time_lin_params_idx = range(prior[2],current[2])
        
        self.batchnorm0 = eqx.experimental.BatchNorm(
                input_size=self.conv_shapes[self.conv_params_idx[0+local_num_shift]][0],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                )
        self.batchnorm1 = eqx.experimental.BatchNorm(
                input_size=self.conv_shapes[self.conv_params_idx[1+local_num_shift]][0],
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.dropout = eqx.nn.Dropout(cfg.parameters.dropout_p,inference=cfg.parameters.inference)


    def forward(self,x_in,embedding,parameters,subkey=None):

        w = parameters[2][0][self.time_lin_params_idx[self.local_num_shift//2]]
        b = parameters[2][1][self.time_lin_params_idx[self.local_num_shift//2]]

        x = vmap(self.batchnorm0,axis_name="batch")(x_in.transpose(0,3,2,1)).transpose(0,3,2,1) 
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
        self.sub_model_num = sub_model_num
        self.local_num_shift = local_num_shift

        prior = param_asso[:sub_model_num].sum(axis=0)
        current = param_asso[:sub_model_num+1].sum(axis=0)
        self.s_lin_params = range(prior[1],current[1])

        self.resnet = resnet(cfg, param_asso, sub_model_num, local_num_shift = self.local_num_shift)

    def forward(self, x_in, embedding, parameters,subkey=None):

        x = self.resnet.forward(x_in, embedding, parameters, subkey=subkey)
        w = parameters[1][0][self.s_lin_params[0+self.local_num_shift//2]]
        b = parameters[1][1][self.s_lin_params[0+self.local_num_shift//2]]

        x_in = jnp.einsum('bhwc,cC->bhwC', x, w) + b
        return x+x_in

class attention():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0) -> None:
        self.sub_model_num = sub_model_num
        self.local_num_shift = local_num_shift
        
        self.attn_shapes = cfg.parameters.attention_linear

        prior = param_asso[:sub_model_num].sum(axis=0)
        current = param_asso[:sub_model_num+1].sum(axis=0)

        self.attn_lin_params_idx = range(prior[3],current[3])


        self.batchnorm0 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=self.attn_shapes[self.attn_lin_params_idx[0+local_num_shift]][0],
                axis_name="batch",
                momentum=cfg.parameters.momentum,
                eps=cfg.parameters.eps,
                # channelwise_affine=True
                )

    def forward(self,x_in,parameters):
        # Get shape for reshapes later
        B, H, W, C = x_in.shape

        # will be replaces with parameters
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

        x = jnp.einsum('bhwHW,bHWc->bhwc', sdpa, v)
        x = jnp.einsum('bhwc,cC->bhwC', x, w4)+b4

        return x+x_in

class down_resnet():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0,maxpool_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = local_num_shift+0)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = local_num_shift+2)
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor)

    def forward(self, x_in, embedding, parameters, subkey = None):
        # split randomness key
        if key is not None:
            subkey = random.split(key*self.sub_model_num,2)

        x0 = self.resnet0.forward(x_in, embedding, parameters,subkey = subkey[0])
        x1 = self.resnet1.forward(x0, embedding, parameters,subkey = subkey[1])
        x2 = vmap(self.maxpool2d,axis_name="batch")(x1.transpose(0,3,2,1)).transpose(0,3,2,1)
        return x0,x1,x2

class down_resnet_attn():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0,maxpool_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 2+local_num_shift)
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor)
        self.attn0 = attention(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.attn1 = attention(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)

    def forward(self, x_in, embedding, parameters, subkey = None):

        # split randomness key
        if key is not None:
            subkey = random.split(key*self.sub_model_num,2)

        x00 = self.resnet0.forward(x_in, embedding, parameters,subkey=subkey[0])
        x01 = self.attn0.forward(x00,parameters)
        x10 = self.resnet1.forward(x01, embedding, parameters,subkey=subkey[1])
        x11 = self.attn1.forward(x10,parameters)
        x2 = vmap(self.maxpool2d,axis_name="batch")(x11.transpose(0,3,2,1)).transpose(0,3,2,1)
        return x01,x11,x2

def upsample2d(x, factor=2):
    # stolen from https://github.com/yang-song/score_sde/blob/main/models/up_or_down_sampling.py
    B, H, W, C = x.shape
    x = jnp.reshape(x, [-1, H, 1, W, 1, C])
    x = jnp.tile(x, [1, 1, factor, 1, factor, 1])
    return jnp.reshape(x, [-1, H * factor, W * factor, C])

class up_resnet():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0,upsampling_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.upsampling_factor = upsampling_factor
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 2+local_num_shift)
        self.resnet2 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)


    def forward(self, x, embedding, x_res0, x_res1, x_res2, parameters, subkey = None):
        
        # split randomness key
        if key is not None:
            subkey = random.split(key*self.sub_model_num,3)

        x = self.resnet0.forward(jnp.concatenate((x,x_res0),axis=-1), embedding, parameters,subkey=subkey[0])
        x = self.resnet1.forward(jnp.concatenate((x,x_res1),axis=-1), embedding, parameters,subkey=subkey[1])
        x = self.resnet2.forward(jnp.concatenate((x,x_res2),axis=-1), embedding, parameters,subkey=subkey[2])
        return x

class up_resnet_attn():
    def __init__(self,cfg, param_asso, sub_model_num, local_num_shift = 0,upsampling_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.upsampling_factor = upsampling_factor
        self.resnet0 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.resnet1 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 2+local_num_shift)
        self.resnet2 = resnet_ff(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)
        self.attn0 = attention(cfg, param_asso,sub_model_num, local_num_shift = 0+local_num_shift)
        self.attn1 = attention(cfg, param_asso,sub_model_num, local_num_shift = 4+local_num_shift)
        self.attn2 = attention(cfg, param_asso,sub_model_num, local_num_shift = 8+local_num_shift)

    def forward(self, x, embedding, x_res0, x_res1, x_res2, parameters, subkey = None):

        # split randomness key
        if key is not None:
            subkey = random.split(key*self.sub_model_num,3)

        x = self.resnet0.forward(jnp.concatenate((x,x_res0),axis=-1), embedding, parameters,subkey=subkey[0])
        x = self.attn0.forward(x,parameters)
        x = self.resnet1.forward(jnp.concatenate((x,x_res1),axis=-1), embedding, parameters,subkey=subkey[1])
        x = self.attn1.forward(x,parameters)
        x = self.resnet2.forward(jnp.concatenate((x,x_res2),axis=-1), embedding, parameters,subkey=subkey[2])
        x = self.attn2.forward(x,parameters)
        return x

class ddpm_unet():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        conv_shapes = cfg.parameters.conv_channels

        param_asso = jnp.array(self.cfg.parameters.model_parameter_association)

        # down
        self.resnet_1 = down_resnet(cfg, param_asso,sub_model_num=1,maxpool_factor=2)
        self.resnet_attn_2 = down_resnet_attn(cfg, param_asso,sub_model_num=2,maxpool_factor=2)
        self.resnet_3 = down_resnet(cfg, param_asso,sub_model_num=3,maxpool_factor=2)
        self.resnet_4 = down_resnet(cfg, param_asso,sub_model_num=4,maxpool_factor=1) # no downsampling here

        # middle
        self.resnet_5 = resnet_ff(cfg, param_asso,sub_model_num=5)
        self.attn_6 = attention(cfg, param_asso,sub_model_num=6)
        self.resnet_7 = resnet_ff(cfg, param_asso,sub_model_num=7)

        # up
        self.resnet_8 = up_resnet(cfg, param_asso,sub_model_num=8)
        self.resnet_9 = up_resnet(cfg, param_asso,sub_model_num=9)
        self.resnet_attn_10 = up_resnet_attn(cfg, param_asso,sub_model_num=10)
        self.resnet_11 = up_resnet(cfg, param_asso,sub_model_num=11)

        # end
        self.batchnorm_12 = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=conv_shapes[-1][0], # Its input is equal to last conv input, as this doesnt change shape
                axis_name="batch",
                momentum=cfg.parameters.momentum,
                eps=cfg.parameters.eps,
                # channelwise_affine=True
                )
        self.upsampling_factor = self.cfg.parameters.upsampling_factor

    def forward(self, x_in, timesteps, parameters, key = None):

        # split key
        if key is not None:
            key, *subkey = random.split(key,13)

        # Timestep embedding
        em_w1 = parameters[2][0][-2]
        em_w2 = parameters[2][0][-1]
        em_b1 = parameters[2][1][-2]
        em_b2 = parameters[2][1][-1]
        embedding_dims = self.cfg.parameters.time_embedding_dims
        embedding = self.get_timestep_embedding(timesteps, embedding_dim = embedding_dims) # embedding -> dense -> nonlin -> dense (Shape = Bx512)
        embedding = jnp.matmul(embedding,em_w1)+em_b1 # 128 -> 512
        embedding = nn.relu(embedding)
        embedding = jnp.matmul(embedding,em_w2)+em_b2 # 512 -> 512

        # down
        print(x_in.shape)
        d0 = lax.conv_general_dilated( 
                lhs = x_in,    
                rhs = parameters[0][0], # kernel is the conv [0] and the first entry i this [0]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )
        print(d0.shape)
        d10,d11,d12 = self.resnet_1.forward(       d0, embedding, parameters, subkey = subkey[1]) # 32x32 -> 16x16     C_out = 128
        d20,d21,d22 = self.resnet_attn_2.forward(  d12, embedding, parameters, subkey = subkey[2]) # 16x16 -> 8x8      C_out = 256
        d30,d31,d32 = self.resnet_3.forward(       d22, embedding, parameters, subkey = subkey[3]) # 8x8 -> 4x4        C_out = 512
        d40,d41,_   = self.resnet_4.forward(       d32, embedding, parameters, subkey = subkey[4]) # 4x4 -> 4x4        C_out = 512

        # middle
        m = self.resnet_5.forward(                 d41, embedding, parameters,subkey = subkey[5]) # 4x4 -> 4x4
        m = self.attn_6.forward(                   m, parameters) # 4x4 -> 4x4
        m = self.resnet_7.forward(                 m, embedding, parameters,subkey = subkey[7]) # 4x4 -> 4x4   C_out = 512

        # up
        u = self.resnet_8.forward(          m, embedding, x_res0=d41, x_res1=d40, x_res2=d32, parameters=parameters, subkey = subkey[8]) # 4x4 -> 4x4   C_out = 512
        u = upsample2d(                     u, factor=self.upsampling_factor) # 4x4 -> 8x8
        u = self.resnet_9.forward(          u, embedding, x_res0=d31, x_res1=d30, x_res2=d22, parameters=parameters, subkey = subkey[9]) # 8x8 -> 8x8   C_out = 512
        u = upsample2d(                     u, factor=self.upsampling_factor) # 8x8 -> 16x16
        u = self.resnet_attn_10.forward(    u, embedding, x_res0=d21, x_res1=d20, x_res2=d12, parameters=parameters, subkey = subkey[10]) # 16x16 -> 16x16 C_out = 256
        u = upsample2d(                     u, factor=self.upsampling_factor) # 16x16 -> 32x32
        u = self.resnet_11.forward(         u, embedding, x_res0=d11, x_res1=d10, x_res2=d0, parameters=parameters, subkey = subkey[11]) # 32x32 -> 32x32 C_out = 128

        # end
        e = vmap(self.batchnorm_12,axis_name="batch")(u.transpose(0,3,2,1)).transpose(0,3,2,1)
        e = nn.relu(e)
        e = lax.conv_general_dilated( 
                lhs = e,    
                rhs = parameters[0][-1], # kernel is the conv [0] and the last entry of these [-1]
                window_strides = [1,1], 
                padding = 'same',
                dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
                )
        return e

    def get_parameters(self,cfg):
        key = random.PRNGKey(cfg.model.key)

        # Get stuff from config
        conv_channels = cfg.model.parameters.conv_channels
        kernel_sizes = cfg.model.parameters.kernel_sizes
        skip_linear = cfg.model.parameters.skip_linear
        time_embed_linear = cfg.model.parameters.time_embed_linear
        attention_linear = cfg.model.parameters.attention_linear
        embedding_parameters = cfg.model.parameters.embedding_parameters

        parameters = [[], [[],[]], [[],[]], [[],[]]] 
        # List of  [Conv, [sL,sB], [eL,eB], [aL,aB]], 
        # L = Linear, B = Bias
        # s = skip_linear, e = time_embedding_linear, a = attention_linear

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
        return jnp.sum(output)

    def loss_fn(self, parameters, true_data, timestep, key = None):
        output = self.forward(true_data, timestep, parameters, key = key)
        loss = self.loss_mse(output, true_data)
        return loss

    def get_timestep_embedding(self, timesteps, embedding_dim: int):
        """
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
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

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
# if __name__ == "__main__":
# Andreas needs
import sys
sys.path.append("/media/sf_Bsc-Diffusion")
# need ends

from utils.utils import get_hydra_config
cfg = get_hydra_config()
# print(cfg.model)

key = random.PRNGKey(69)

img = jnp.ones((
        cfg.model.parameters.batch_size,    # Batchsize
        cfg.model.parameters.img_h,         # h
        cfg.model.parameters.img_w,         # w
        cfg.model.parameters.conv_channels[0][0],# channels
        ),dtype=jnp.float32)
img = img.at[0, 0, 2:2+10, 2:2+10].set(0.0) 
B, H, W, C = img.shape

model = ddpm_unet(cfg.model)
parameters, key = model.get_parameters(cfg)
get_grad = grad(jit(model.loss_fn),0)
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

    print("If no 'False' appears then it all worked out")


check_grads_beq_zero(grads)
