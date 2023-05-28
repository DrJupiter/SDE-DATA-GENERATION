#%%
import matplotlib.pyplot as plt
import numpy as np
#%%
fig, ax = plt.subplots()
ax.bar(["fisk","2","ged",",","1","22","3"],[1,2,3,0,5,2,2])



# %%

no_s_m = [3.024e-4,1.377e-2,5.417e-2]
no_s_std = [4.713e-5,1.090e-3,4.277e-4]

shard_512_m = {"B_d1":2.696e-4,"B_d2":2.754e-4,"  ":0,"H_d1":3.391e-4,"H_d2":2.753e-4," ":0,"W_d1":2.720e-4,"W_d2":2.745e-4,"   ":0,"C_d1":9.466e-3,"C_d2":5.403e-3}
shard_512_std = {"B_d1":4.571e-5,"B_d2":4.249e-5,"  ":0,"H_d1":3.093e-5,"H_d2":2.018e-5," ":0,"W_d1":6.007e-5,"W_d2":4.072e-5,"   ":0,"C_d1":7.661e-4,"C_d2":4.453e-4}

shard_4k_m = []

shard_8k_m = []

#%%

fig, ax = plt.subplots()
ax.bar(shard_512_m.keys(),shard_512_m.values(),yerr=shard_512_std.values())
# ax.set_yscale("log")

# %%
