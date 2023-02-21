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
    def __init__(self,cfg) -> None:
        self.maxpool2d = eqx.nn.MaxPool2d(2,2) # add to config
        self.batchnorm1bn = eqx.experimental.BatchNorm(  # Make 1 for each needed, as they have differetent input shapes
                input_size=cfg.parameters.channels[1][1], #takes in-channels as input_size
                axis_name="batch",
                momentum=0.99,
                eps=1e-05,
                # channelwise_affine=True
                ) 
        self.strides = cfg.parameters.kernel_stride

    def forward(self, input_x, parameters):
        conv10 = lax.conv( # vmap on these as well? or are they already? I assume they already are
                lhs = input_x,    # lhs = NCHW image tensor
                rhs = parameters[0], # rhs = OIHW conv kernel tensor
                window_strides = self.strides[0],  # window strides
                padding = 'same'
                )
        conv10r = nn.relu(conv10)
        conv11 = lax.conv(
                lhs = conv10r,    
                rhs = parameters[1], 
                window_strides = self.strides[1],  
                padding = 'same'
                )
        conv11r = nn.relu(conv11)
        # print(conv11r.shape)
        conv1bn = vmap(self.batchnorm1bn,axis_name="batch")(conv11r) 
        conv1mp = vmap(self.maxpool2d,axis_name="batch")(conv1bn)

        return conv1mp



class unet():
    def __init__(self,cfg) -> None:
        print(cfg)
        self.down1 = down(cfg)

    def get_parameters(self,cfg):
        key = random.PRNGKey(cfg.model.key)

        channels = cfg.model.parameters.channels
        kernel_sizes = cfg.model.parameters.kernel_sizes

        key, *subkey = random.split(key,cfg.model.parameters.N_channels+1)
        parameters = []
        for i,((in_channel,out_channel),(kernel_size_h,kernel_size_w)) in enumerate(zip(channels,kernel_sizes)): 
            parameters.append(random.normal(subkey[i], ((out_channel,in_channel,kernel_size_h,kernel_size_w)), dtype=jnp.float32))
        return parameters

    def forward(self,input_img,parameters):
        out = self.down1.forward(input_img, parameters)
        # print(out.shape)
        return out
        
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
    loss_fn = jit(model.loss_fn)
    print("loss",loss_fn(parameters, img))
    print("parameter1 shape",get_grad(parameters, img)[0].shape)
    print("parameter2 shape",get_grad(parameters, img)[1].shape)

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