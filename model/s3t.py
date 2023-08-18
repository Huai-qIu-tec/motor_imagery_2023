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
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (12, 10), stride=(1, 10)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
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

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=4, n_classes=3, **kwargs):
        super().__init__(
            # channel_attention(),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(500),
                    channel_attention(),
                    nn.Dropout(0.5),
                )
            ),

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class channel_attention(nn.Module):
    def __init__(self, sequence_num=500, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation
        self.channel_dim = 12
        self.query = nn.Sequential(
            nn.Linear(self.channel_dim, self.channel_dim),
            nn.LayerNorm(self.channel_dim),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(self.channel_dim, self.channel_dim),
            # nn.LeakyReLU(),
            nn.LayerNorm(self.channel_dim),
            nn.Dropout(0.3)
        )

        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(self.channel_dim, self.channel_dim),
            # nn.LeakyReLU(),
            nn.LayerNorm(self.channel_dim),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = self.pooling(temp_query)
        channel_key = self.pooling(temp_key)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


class Recognize:
    def __init__(self, net, filterA, filterB):
        self.net = net
        self.filterA = filterA
        self.filterB = filterB

    def predict(self, data, personID=1):
        # print(data.shape)
        shape = data.shape[1]
        eeg_data = np.zeros((20, 500))
        # <=2s时
        if shape <= 500:
            eeg_data[:, :shape] = data
        # 3s时
        if 500 < shape <= 750:
            eeg_data[:, 0:shape - 500] = data[:, 500:]
            eeg_data[:, shape - 500:] = np.flip(data[:, 500:], axis=1)
        # 4s时
        if 750 < shape <= 1000:
            eeg_data[:, 0:shape - 750] = data[:, 750:]
            eeg_data[:, shape - 750:] = np.flip(data[:, 750:], axis=1)

        filter_data = signal.filtfilt(self.filterB, self.filterA, eeg_data)
        filter_data = (filter_data - np.mean(filter_data, axis=1, keepdims=True)) \
                      / (np.std(filter_data, axis=1, keepdims=True) + 1e-8)

        root_dir = os.path.dirname(os.path.abspath(__file__))
        path = root_dir + '/checkpoints/ViT_S{}.pth'.format(personID)
        Wb_path = root_dir + '/checkpoints/S{}_trainWb.pkl'.format(personID)
        with open(Wb_path, 'rb') as f:
            Wb = pickle.load(f)
        filter_data = np.einsum('bcd, ce -> bed', np.expand_dims(filter_data, axis=0), Wb)

        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']
        self.net.load_state_dict(model_state_dict)

        outputs = self.net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        result = torch.argmax(outputs, dim=1)[0]

        return result.item() + 1
