import torch
from torch import nn
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import OrderedDict
from einops import rearrange
from model_tfm import PerceiverEncoder


class VideoCaptionModel(nn.Module):
    def __init__(self,
                 num_latents: int = 10, 
                 num_layers: int = 2, 
                 prefix_size: int = 512,
                 use_context_perceiver: int = 0,
                 use_subtitle_perceiver: int = 0,
                 **kwargs,
                 ):
        super().__init__()
        if len(kwargs):
            print(f'WARNING [VideoCaptionModel] kwargs not used: {kwargs}')
        self.num_layers = num_layers
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        ### visual ###
        self.perceiver = PerceiverEncoder(
            num_latents=num_latents, 
            d_latents=prefix_size, 
            num_layers=num_layers, 
            nhead=prefix_size//64)
        self.project = nn.Linear(prefix_size, self.gpt_embedding_size)
        nn.init.normal_(self.project.weight, std=prefix_size ** -0.5)
        nn.init.zeros_(self.project.bias)

        ### context ###
        self.use_context_perceiver = use_context_perceiver
        assert use_context_perceiver in [0, 1]
        if use_context_perceiver == 1:
            # produce <BOS>, <EOS> around the context features 
            self.context_special_token = nn.Embedding(2, embedding_dim=self.gpt_embedding_size)

        ### subtitle ###
        self.use_subtitle_perceiver = use_subtitle_perceiver
        assert use_subtitle_perceiver in [0, 3, 4]
        if use_subtitle_perceiver in [3, 4]:
            # produce <BOS>, <EOS> around the context features 
            self.subtitle_special_token = nn.Embedding(2, embedding_dim=self.gpt_embedding_size)

        ### BOS token for AD generation
        self.bos_token = nn.Embedding(1, embedding_dim=self.gpt_embedding_size)

    def wrap_context(self, context_embed, prompt=None):
        """assume context_embed: B,N,C. Add <bos> <eos> on it"""
        assert prompt is None
        B = context_embed.shape[0]
        bos = self.context_special_token.weight[None, 0:1].repeat(B,1,1)
        eos = self.context_special_token.weight[None, 1:2].repeat(B,1,1)
        return torch.cat((bos, context_embed, eos), dim=1)

    def wrap_subtitle(self, subtitle_embed):
        B = subtitle_embed.shape[0]
        """assume subtitle_embed: B,N,C. Add <bos> <eos> on it"""
        bos = self.subtitle_special_token.weight[None, 0:1].repeat(B,1,1)
        eos = self.subtitle_special_token.weight[None, 1:2].repeat(B,1,1)
        return torch.cat((bos, subtitle_embed, eos), dim=1)

    def forward(self, visual_feature, mask=None, labels=None):
        """purely for visual prompt"""
        # visual_feature: b t c
        # prefix_vector: b k c
        latent_vector = self.perceiver(visual_feature)
        prefix_vector = self.project(latent_vector)
        return prefix_vector



if __name__ == '__main__':
    # UNIT TEST
    from gpt_utils import generate_beam, generate_greedy
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = VideoCaptionModel()
    prefix_vector = model(torch.randn(1, 1, 512))
    print(generate_greedy(model, tokenizer, embed=prefix_vector))
    print(generate_beam(model, tokenizer, embed=prefix_vector))