#%%
# computations
import jax
import jax.numpy as jnp
import jax.lax as lax

# plot
import numpy as np
import matplotlib.pyplot as plt
#%%

### Jax input denotions
# N - batch dimension
# H - spatial height
# W - spatial height
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension

### CONV
# conv = lax.conv(
#             img,      # NCHW image tensor
#             kernel,   # OIHW conv kernel tensor
#             (1, 1),   # window strides
#             'SAME')   # padding mode

# %%

### MaxPooling


# %%

### ReLU
# %%

### BatchNorm

# %%

### DropOut?