#%%
import matplotlib.pyplot as plt

# %%

no_s_m = [3.024e-4,1.377e-2,5.417e-2]
no_s_std = [4.713e-5,1.090e-3,4.277e-4]

shard_512_m = {"no shard":no_s_m[0],"    ":0,"B_d1":2.696e-4,"B_d2":2.754e-4,"  ":0,"H_d1":3.391e-4,"H_d2":2.753e-4," ":0,"W_d1":2.720e-4,"W_d2":2.745e-4,"   ":0,"C_d1":9.466e-3,"C_d2":5.403e-3}
shard_512_std = {"no shard":no_s_std[0],"    ":0,"B_d1":4.571e-5,"B_d2":4.249e-5,"  ":0,"H_d1":3.093e-5,"H_d2":2.018e-5," ":0,"W_d1":6.007e-5,"W_d2":4.072e-5,"   ":0,"C_d1":7.661e-4,"C_d2":4.453e-4}

shard_4k_m = {"no shard":no_s_m[1],"    ":0,"B_d1":1.194e-2,"B_d2":1.164e-2," ":0,"H_d1":1.209e-2,"H_d2":1.175e-2,"  ":0,"W_d1":1.214e-2,"W_d2":1.184e-2,"   ":0, "C_d1":7.935e-2,"C_d2":4.700e-2}
shard_4k_std = {"no shard":no_s_std[1],"    ":0,"B_d1":9.586e-4,"B_d2":9.312e-4," ":0,"H_d1":9.697e-4,"H_d2":9.322e-4,"  ":0,"W_d1":9.691e-4,"W_d2":9.414e-4,"   ":0, "C_d1":6.343e-3,"C_d2":3.802e-3}

shard_8k_m = {"no shard":no_s_m[2],"    ":0,"B_d1":3.764e-2,"B_d2":4.816e-2," ":0,"H_d1":4.739e-2,"H_d2":4.819e-2,"  ":0,"W_d1":4.768e-2,"W_d2":5.815e-2,"   ":0, "C_d1":1.715e-1,"C_d2":1.121e-1}
shard_8k_std = {"no shard":no_s_std[2],"    ":0,"B_d1":3.870e-3,"B_d2":3.831e-3," ":0,"H_d1":3.775e-3,"H_d2":3.814e-3,"  ":0,"W_d1":3.759e-3,"W_d2":3.821e-3,"   ":0, "C_d1":1.371e-2,"C_d2":8.946e-3}

colours = ["blue","black","red","green","black","red","green","black","red","green","black","red","green"]

#%%

k = None # -3 # None

fig, axs = plt.subplots(3,sharex='all')
fig.suptitle('Compuation speeds of different sharding configurations on 2 A100s', fontsize=16, x=0.5,y=0.93)

axs[0].set_title("C=512")
axs[0].set_ylabel("seconds")
axs[0].set_xlabel("Sharding confirguration")
axs[0].bar(list(shard_512_m.keys())[:k],list(shard_512_m.values())[:k],yerr=list(shard_512_std.values())[:k],color  = colours[:k])
axs[0].axhline(y = min([x for x in list(shard_512_m.values())[:k] if x>0]), color = 'black', linestyle = '--')

axs[1].set_title("C=4k")
axs[1].set_ylabel("seconds")
axs[1].set_xlabel("Sharding confirguration")
axs[1].bar(list(shard_4k_m.keys())[:k],list(shard_4k_m.values())[:k],yerr=list(shard_4k_std.values())[:k],color  = colours[:k])
axs[1].axhline(y = min([x for x in list(shard_4k_m.values())[:k] if x>0]), color = 'black', linestyle = '--')

axs[2].set_title("C=8k")
axs[2].set_ylabel("seconds")
axs[2].set_xlabel("Sharding confirguration")
axs[2].bar(list(shard_8k_m.keys())[:k],list(shard_8k_m.values())[:k],yerr=list(shard_8k_std.values())[:k],color  = colours[:k])
axs[2].axhline(y = min([x for x in list(shard_8k_m.values())[:k] if x>0]), color = 'black', linestyle = '--')

fig.set_figwidth(14)
fig.set_figheight(14)




# ax.set_yscale("log")

# %%

for i in range(len(colours[:k])):
    print(list(shard_4k_m.keys())[:k][i],colours[:k][i])

#%%
([x for x in list(shard_512_m.values())[:k] if x>0])


#%%
N = [2.406778e-03,2.028717e-04]
B = [[1.978e-03,1.782e-04], [1.977e-03,1.524e-04]]
H = [[1.353e-02,1.091e-03], [1.350e-02,1.094e-03]]
W = [[1.443e-02,1.162e-03], [1.445e-02,1.166e-03]]
C = [[1.034e-02,8.266e-04], [6.685e-03,5.452e-04]]

data = [B,H,W,C]

d1_m = []
d1_std = []
d2_m = []
d2_std = []

for dim in data:
    for i,point in enumerate(dim):
        mean,std = point
        if i==0:
            d1_m.append(mean)
            d1_std.append(std)
        if i==1:
            d2_m.append(mean)
            d2_std.append(std)

#%%
names = ["B","H","W","C"]
colors = ["b","y","g","r"]

fig, axs = plt.subplots(1,3,sharey="all",gridspec_kw={'width_ratios': [1, 4, 4]})
fig.suptitle('Compuation speeds of different sharding configurations for Convolution on 2 A100s using C=512', fontsize=16, x=0.5,y=0.95)

axs[0].set_title("No sharding")
axs[0].set_ylabel("seconds")
axs[0].set_xlabel("Sharding confirguration")
axs[0].bar(["No sharding"],N[0],yerr=N[1],color = ["magenta"])

axs[1].set_title("D1")
axs[1].set_ylabel("seconds")
axs[1].set_xlabel("Sharding confirguration")
axs[1].bar(names,d1_m,yerr=d1_std,color  = colors)
# axs[0].axhline(y = min([x for x in list(shard_512_m.values())[:k] if x>0]), color = 'black', linestyle = '--')

axs[2].set_title("D2")
axs[2].set_ylabel("seconds")
axs[2].set_xlabel("Sharding confirguration")
axs[2].bar(names,d2_m,yerr=d2_std,color  = colors)
# axs[1].axhline(y = min([x for x in list(shard_4k_m.values())[:k] if x>0]), color = 'black', linestyle = '--')

fig.set_figwidth(14)
fig.set_figheight(8)
# %%
