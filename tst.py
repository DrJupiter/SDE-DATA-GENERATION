import jax.numpy as jnp
from jax import lax
in_dim = 5
img1 = jnp.arange(2*in_dim*9*9).reshape(2,in_dim,9,9)*(1e-3)
k1 = jnp.arange(10*in_dim*3*3).reshape(10,in_dim,3,3)*(1e-3)

# e = lax.conv_general_dilated( 
#             lhs = img1,   
#             rhs = k1, 
#             window_strides = [1,1], 
#             padding = 'same',
#             dimension_numbers = ('NCHW', 'OIHW', 'NCHW')
#             )

img2 = img1.reshape(2,9,9,in_dim)
k2 = k1.reshape(3,3,in_dim,10)
e2 = lax.conv_general_dilated( 
            lhs = img2,    
            rhs = k2, 
            window_strides = [1,1], 
            padding = 'same',
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
            )

# print(jnp.sum(k1),jnp.sum(k2),"\n",jnp.sum(img1),jnp.sum(img2))

print(jnp.sum(e2))
