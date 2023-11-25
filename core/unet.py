import torch
from torch import nn
from utils import default, exists
from functools import partial
from resnet import (
    ResnetBlocWithAttention,
    Block
)
from nn_utils import (
    PositionalEncoding,
    Downsample,
    Upsample,
    Swish
)

class Unet(nn.Module):
    def __init__(
       self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4,),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):

            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)

            channel_mult = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks):
                downs.append(
                    ResnetBlocWithAttention(
                        pre_channel, 
                        channel_mult, 
                        noise_level_emb_dim=noise_level_channel, 
                        norm_groups=norm_groups, 
                        dropout=dropout, 
                        with_attn=use_attn
                    )
                )
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
                
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttention(
                pre_channel, 
                pre_channel, 
                noise_level_emb_dim = noise_level_channel, 
                norm_groups = norm_groups,
                dropout = dropout, 
                with_attn = True
            ),
            ResnetBlocWithAttention(
                pre_channel, 
                pre_channel, 
                noise_level_emb_dim = noise_level_channel, 
                norm_groups = norm_groups,
                dropout = dropout, 
                with_attn = False
            )
        ])

        ups = []

        for ind in reversed(range(num_mults)):

            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]

            for _ in range(0, res_blocks+1):

                ups.append(
                    ResnetBlocWithAttention(
                        pre_channel+feat_channels.pop(), 
                        channel_mult, 
                        noise_level_emb_dim = noise_level_channel, 
                        norm_groups = norm_groups,
                        dropout = dropout, 
                        with_attn = use_attn
                    )
                )
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(
            pre_channel,
            default(
                out_channel, 
                in_channel
            ), 
            groups=norm_groups
        )

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttention):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttention):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttention):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)