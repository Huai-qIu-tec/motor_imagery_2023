import math
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, channels=24, sample=250, dropout=0.5):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, emb_size // 4, (1, 64), (1, 1), padding=(0, 64 // 2)),
            nn.ELU(),
            nn.BatchNorm2d(emb_size // 4),
            nn.AvgPool2d((1, 4)),
            # nn.Dropout(dropout),

            nn.Conv2d(emb_size // 4, emb_size // 2, (1, 16), (1, 1), padding=(0, 16 // 2)),
            nn.ELU(),
            nn.BatchNorm2d(emb_size // 2),
            nn.AvgPool2d((1, 8)),
            # nn.Dropout(dropout),

            nn.Conv2d(emb_size // 2, emb_size, (1, 4), (1, 1), padding=(0, 4 // 2)),
            nn.ELU(),

        )

        x = torch.randn((1, 1, channels, sample))
        out = self.projection(x)

        self.pooling = nn.Sequential(
            nn.AvgPool2d((1, out.shape[-1])),
            Rearrange('b e h w -> b (h w) e'),
            # nn.Dropout(dropout),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = self.pooling(x)

        x = torch.cat([self.cls.repeat(x.shape[0], 1, 1), x], dim=1)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, channels, emb_size):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, channels + 1, emb_size))

    def forward(self, x):
        x = self.pos_emb + x
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.attention_weights = None

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        self.attention_weights = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(self.attention_weights)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, emb_size, fn, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.dropout(self.fn(self.norm(x)))

        return res + x



class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion=2, dropout=0.5):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.feed_forward(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

        self.res_add1 = ResidualAdd(emb_size, self.attention, dropout)
        self.res_add2 = ResidualAdd(emb_size, self.feed_forward, dropout)

    def forward(self, x):
        y = self.res_add1(x)
        y = self.res_add2(y)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads, channels, expansion, dropout):
        super(TransformerEncoder, self).__init__()
        self.positional_embedding = PositionalEmbedding(channels, emb_size)
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth

        for i in range(depth):
            self.blks.add_module("block" + str(i), TransformerEncoderBlock(emb_size, num_heads, expansion, dropout))

    def forward(self, x):
        x = self.positional_embedding(x)
        for i, blk in enumerate(self.blks):
            x = blk(x)
            self.attention_weights[i] = blk.attention.attention_weights

        return x


class ClassifyHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super(ClassifyHead, self).__init__()
        self.n_classes = n_classes

        self.fc = nn.Sequential(
            nn.Linear(emb_size, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        cls_head = x[:, 0]

        return self.fc(cls_head)

class Conformer(nn.Module):
    def __init__(self, emb_size=128, depth=4, num_heads=4, channels=24, samples=250, expansion=2, dropout=0.5, n_classes=3):
        super(Conformer, self).__init__()
        self.patch_embedding = PatchEmbedding(emb_size, channels, samples)
        self.transformer = TransformerEncoder(depth, emb_size, num_heads, channels, expansion, dropout)
        self.cls = ClassifyHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.cls(x)

        return x