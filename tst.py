#%%
import jax.numpy as jnp
from jax import lax
in_dim = 2
img1 = jnp.arange(3*in_dim*32*32).reshape(3,in_dim,32,32)
k1 = jnp.arange(10*in_dim*3*3).reshape(10,in_dim,3,3)

# e = lax.conv_general_dilated( 
#             lhs = img1,   
#             rhs = k1, 
#             window_strides = [1,1], 
#             padding = 'same',
#             dimension_numbers = ('NCHW', 'OIHW', 'NCHW')
#             )

img2 = img1.transpose(0,2,3,1)
k2 = k1.transpose(2,3,1,0)
e2 = lax.conv_general_dilated( 
            lhs = img2,    
            rhs = k2, 
            window_strides = [1,1], 
            padding = 'same',
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
            )

# print("IMG1\n",img1)
# print("IMG2\n",img2)
print(img2.shape)
print(k2.shape)
print(jnp.sum(k1),jnp.sum(k2),"\n",jnp.sum(img1),jnp.sum(img2))
print("NCHW:",jnp.sum(e),"\nNHWC",jnp.sum(e2))
#%%

img1.shape
# 1,3,3,in_dim
print(img1.transpose(0,2,3,1))
img2

#%%
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

key = jr.PRNGKey(0)
mkey, dkey = jr.split(key)
model = eqx.nn.Sequential([
    eqx.experimental.BatchNorm(input_size=4, axis_name="batch"),
])

x = jr.normal(dkey, (2,32,32,4))
jax.vmap(model, axis_name="batch")(x.transpose(0,3,2,1)).transpose(0,3,2,1).shape
# BatchNorm will automatically update its running statistics internally.
# %%
