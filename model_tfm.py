"""
Transformer part modified from OpenAI's CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
Caption module modified from ClipCap: https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing#scrollTo=OArDkm_24w4L 

Designed for short video captioning. 
"""

import torch
from torch import nn
from typing import Tuple, List, Union, Optional
from collections import OrderedDict
from torch.nn import LayerNorm
from einops import rearrange


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock_Step(nn.Module):
    def __init__(self, d_model: int, n_head: int,):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, pos: torch.Tensor = None):
        key_padding_mask = key_padding_mask.to(device=x.device) if key_padding_mask is not None else None
        q = k = self.with_pos_embed(x, pos)
        return self.attn(q, k, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, pos: torch.Tensor = None):
        x_norm = self.ln_1(x)
        x = x + self.attention(x_norm, key_padding_mask=key_padding_mask, pos=pos)
        x = x + self.mlp(self.ln_2(x))
        return x, x_norm


class TemporalEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock_Step(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, pos: torch.Tensor = None):
        intermediate = []
        for block in self.resblocks:
            x, x_norm = block(x, key_padding_mask, pos)
            intermediate.append(x_norm)
        intermediate.pop(0)
        intermediate.append(x)
        return intermediate


class PerceiverEncoder(nn.Module):
    """Perceiver-like module, with TransformerEncoder([latent; features])"""
    def __init__(self, num_latents=16, d_latents=768, nhead=8, num_layers=2):
        super().__init__()
        self.num_latents = num_latents
        self.latent = nn.Parameter(torch.empty(num_latents, d_latents))
        self.temporal_pos_embed = nn.Parameter(torch.empty(512, d_latents))
        self.encoder = TemporalEncoder(width=d_latents, layers=num_layers, heads=nhead)
        self.visual_prenorm = LayerNorm(d_latents)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.latent, mean=0, std=1)
        nn.init.normal_(self.temporal_pos_embed, mean=0, std=1.0)
        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5
        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, visual_feature, key_padding_mask=None):
        B, T, *_ = visual_feature.shape
        visual_feature = rearrange(visual_feature, 'b t c -> t b c')
        temp_pos = self.temporal_pos_embed[0:T, None, :]
        visual_feature = self.visual_prenorm(visual_feature) + temp_pos
        latent = self.latent[:,None,:].repeat(1,B,1)  # k,b,c
        concat = torch.cat((latent, visual_feature), dim=0)
        enc_out = self.encoder(concat, key_padding_mask, pos=None)[-1]  # last layer output

        latent_out = enc_out[0:self.num_latents, :]
        return rearrange(latent_out, 'k b c -> b k c')
