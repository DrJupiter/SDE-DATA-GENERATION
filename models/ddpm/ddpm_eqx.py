import equinox as eqx
import jax

from models.ddpm.building_blocks.ddpm_eqx_blocks import time_embed, resnet_ff, attention, up_resnet, up_resnet_attn, down_resnet, down_resnet_attn

class ddpm(eqx.Module):
    conv_layers: list
    down_layers: list
    # mid_layers: list
    up_layers: list
    time_embed_layers: list

    def __init__(self, cfg, key) -> None:
        key, *subkey = jax.random.split(key, 14)
        
        self.time_embed_layers = [time_embed(32, 128, key=subkey[12])]

        conv0 = eqx.nn.Conv(num_spatial_dims = 2, key = subkey[0], kernel_size=[3,3],     in_channels = 3, out_channels = 32) # conv_layers
        conv11 = eqx.nn.Conv(num_spatial_dims = 2, key = subkey[1], kernel_size=[3,3],      in_channels = 32, out_channels = 3) # conv_layers
        self.conv_layers = [conv0,conv11]

        d_resnet1 = down_resnet(cfg, key=subkey[2], maxpool_factor=2, in_channel= 32*(32**2), out_channel=64*(32**2), embedding_dim=128)
        d_a_resnet2 = down_resnet_attn(cfg, key=subkey[3], maxpool_factor=1, in_channel= 64*(16**2), out_channel=64*(16**2), embedding_dim=128)
        # d_resnet3 = down_resnet(cfg, key=subkey[4], maxpool_factor=2, in_channel= 32, out_channel=64, embedding_dim=128)
        self.down_layers = [d_resnet1,d_a_resnet2]

        # u_resnet9 = up_resnet(cfg, subkey[11], maxpool_factor=1, in_channel=64, out_channel=32, embedding_dim=128)
        u_a_resnet10 = up_resnet_attn(cfg, key=subkey[10], maxpool_factor=2, in_channel=64, out_channel=64, embedding_dim=128)
        u_resnet11 = up_resnet(cfg, key=subkey[11], maxpool_factor=1, in_channel=64, out_channel=32, embedding_dim=128)
        self.up_layers = [u_a_resnet10,u_resnet11]


    def __call__(self, x, timesteps, key):
        
        key, *subkey = jax.random.split(key, 13)

        embed = self.time_embed_layers[0](timesteps, embedding_dims=32)

        # start
        x_32_0 = self.conv_layers[0](x)
        
        # Down
        x_32_1, x_32_2, x_16_0 = self.down_layers[0](x_32_0, embedding=embed, parameters=None, subkey = subkey[0])
        x_16_1, x_16_2, x = self.down_layers[1](x_16_0, embedding=embed, parameters=None, subkey = subkey[0])

        # Mid
        # x = self.mid_layers[0](x, embedding=embed, parameters=None, subkey = subkey[0])

        # Up
        x = self.up_layers[0](x, x_16_2, x_16_1, x_16_0, embedding=embed, parameters=None, subkey = subkey[0])
        x = self.up_layers[1](x, x_32_2, x_32_1, x_32_0, embedding=embed, parameters=None, subkey = subkey[0])

        # End
        x = self.conv_layers[1](x)

        return x
