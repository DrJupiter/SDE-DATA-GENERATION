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


class down():
    def __init__(self,cfg,sub_model_num,maxpool_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.maxpool2d = eqx.nn.MaxPool2d(maxpool_factor,maxpool_factor) # add to config
        self.batchnorm1bn = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num+0][1], #takes in-channels as input_size
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.batchnorm2bn = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num+1][1], #takes in-channels as input_size
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.strides = cfg.parameters.kernel_stride[2*self.sub_model_num:2*self.sub_model_num+2]

    def forward(self, input_x, parameters):
        conv1 = lax.conv( # vmap on these as well? or are they already? I assume they already are
                lhs = input_x,    # lhs = NCHW image tensor
                rhs = parameters[2*self.sub_model_num], # rhs = OIHW conv kernel tensor
                window_strides = self.strides[0],  # window strides
                padding = 'same'
                )
        conv1r = nn.relu(conv1) # vmap on these?
        conv1bn = vmap(self.batchnorm1bn,axis_name="batch")(conv1r) 

        conv2 = lax.conv(
                lhs = conv1bn,    
                rhs = parameters[2*self.sub_model_num+1], 
                window_strides = self.strides[1],  
                padding = 'same'
                )
        conv2r = nn.relu(conv2) # vmap on these?
        conv2bn = vmap(self.batchnorm2bn,axis_name="batch")(conv2r) 
        conv2mp = vmap(self.maxpool2d,axis_name="batch")(conv2bn)
        return conv2mp, conv2r

def upsample2d(x, factor=2):
    # stolen from https://github.com/yang-song/score_sde/blob/main/models/up_or_down_sampling.py
    _N, C, H, W = x.shape
    x = jnp.reshape(x, [-1, C, 1, H, 1, W])
    x = jnp.tile(x, [1, 1, factor, 1, factor, 1])
    return jnp.reshape(x, [-1, C, H * factor, W * factor])

class up():
    def __init__(self,cfg,sub_model_num,upsampling_factor=2) -> None:
        self.sub_model_num = sub_model_num
        self.upsampling_factor = upsampling_factor
        self.batchnorm1bn = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num][1], #takes in-channels as input_size
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.batchnorm2bn = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[2*sub_model_num+1][1], #takes in-channels as input_size
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.strides = cfg.parameters.kernel_stride[2*self.sub_model_num:2*self.sub_model_num+2]

    def forward(self, residual_x, input_x, parameters):
        # upsample img to high img resolution
        upsampled = upsample2d(input_x, factor=self.upsampling_factor)

        # concat the channels
        catted = jnp.concatenate((residual_x,upsampled),axis=1)

        # Apply the convolution etc.
        conv1 = lax.conv( 
                lhs = catted,    # lhs = NCHW image tensor
                rhs = parameters[2*self.sub_model_num], # rhs = OIHW conv kernel tensor
                window_strides = self.strides[0],  # window strides
                padding = 'same'
                )
        conv1r = nn.relu(conv1)
        conv1bn = vmap(self.batchnorm1bn,axis_name="batch")(conv1r) 

        conv2 = lax.conv(
                lhs = conv1bn,    
                rhs = parameters[2*self.sub_model_num+1], 
                window_strides = self.strides[1],  
                padding = 'same'
                )
        conv2r = nn.relu(conv2)
        conv2bn = vmap(self.batchnorm2bn,axis_name="batch")(conv2r) 
        return conv2bn

class unet():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.down1 = down(cfg,sub_model_num=0,maxpool_factor=2)
        self.down2 = down(cfg,sub_model_num=1,maxpool_factor=2)
        self.down3 = down(cfg,sub_model_num=2,maxpool_factor=2)
        self.down4 = down(cfg,sub_model_num=3,maxpool_factor=2)
        self.down5 = down(cfg,sub_model_num=4,maxpool_factor=1) # they dont downscale the last time. And this is effectivly the same as not downscaling. A little slower tho, so dont have this in the final
        self.up1 = up(cfg,sub_model_num=5,upsampling_factor=2)
        self.up2 = up(cfg,sub_model_num=6,upsampling_factor=2)
        self.up3 = up(cfg,sub_model_num=7,upsampling_factor=2)
        self.up4 = up(cfg,sub_model_num=8,upsampling_factor=2)
        # self.up5 = up(cfg,sub_model_num=9,upsampling_factor=1) # same as before

    def forward(self,input_img,parameters):
        out1, pre_max_out1 = self.down1.forward(input_img, parameters)
        out2, pre_max_out2 = self.down2.forward(out1, parameters)
        out3, pre_max_out3 = self.down3.forward(out2, parameters)
        out4, pre_max_out4 = self.down4.forward(out3, parameters)
        out5, _ = self.down5.forward(out4, parameters)

        out6 = self.up1.forward(pre_max_out4, out5, parameters)
        out7 = self.up2.forward(pre_max_out3, out6, parameters)
        out8 = self.up3.forward(pre_max_out2, out7, parameters)
        out9 = self.up4.forward(pre_max_out1, out8, parameters)

        out10 = lax.conv(
                lhs = out9,    
                rhs = parameters[-1], 
                window_strides = self.cfg.parameters.kernel_stride[-1],  
                padding = 'same'
                )

        print("final shape ==",out10.shape)
        return nn.sigmoid(out10)

    def get_parameters(self,cfg):
        key = random.PRNGKey(cfg.model.key)

        channels = cfg.model.parameters.channels
        kernel_sizes = cfg.model.parameters.kernel_sizes

        key, *subkey = random.split(key,len(cfg.model.parameters.channels)+1)
        parameters = []
        for i,((in_channel,out_channel),(kernel_size_h,kernel_size_w)) in enumerate(zip(channels,kernel_sizes)): 
            parameters.append(random.normal(subkey[i], ((out_channel,in_channel,kernel_size_h,kernel_size_w)), dtype=jnp.float32))
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

    from utils.utility import get_hydra_config
    cfg = get_hydra_config()
    # print(cfg.model)

    img = jnp.zeros((
            cfg.model.parameters.batch_size,    # Batchsize
            cfg.model.parameters.channels[0][0],# channels
            cfg.model.parameters.img_h,         # h
            cfg.model.parameters.img_w),        # w
                dtype=jnp.float32)
    img = img.at[0, 0, 2:2+10, 2:2+10].set(1.0) 

    model = unet(cfg.model)
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


#%&
#%%
# def build_unet_model(input_layer, start_neurons):
#     # contracting path (left side) 
#     conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
#     conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
#     pool1 = MaxPooling2D((2, 2))(conv1)
#     pool1 = Dropout(0.25)(pool1)

#     conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
#     conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
#     pool2 = MaxPooling2D((2, 2))(conv2)
#     pool2 = Dropout(0.5)(pool2)

#     conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
#     conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
#     pool3 = MaxPooling2D((2, 2))(conv3)
#     pool3 = Dropout(0.5)(pool3)

#     conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
#     conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
#     pool4 = MaxPooling2D((2, 2))(conv4)
#     pool4 = Dropout(0.5)(pool4)

#     # Middle
#     convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
#     convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
#     # expansive path (right side)
#     deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
#     uconv4 = concatenate([deconv4, conv4])
#     uconv4 = Dropout(0.5)(uconv4)
#     uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
#     uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

#     deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
#     uconv3 = concatenate([deconv3, conv3])
#     uconv3 = Dropout(0.5)(uconv3)
#     uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
#     uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

#     deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
#     uconv2 = concatenate([deconv2, conv2])
#     uconv2 = Dropout(0.5)(uconv2)
#     uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
#     uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

#     deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#     uconv1 = concatenate([deconv1, conv1])
#     uconv1 = Dropout(0.5)(uconv1)
#     uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
#     uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
#     output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
#     return output_layer

# input_layer = (32, 32, 1)
# output_layer = build_model(input_layer, 16)