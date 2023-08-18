import pickle
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pywt


def softmax(X, dim=-1):
    X_max = torch.max(X, dim=dim, keepdim=True)[0]
    X_exp = torch.exp(X - X_max)

    partition = X_exp.sum(dim=dim, keepdim=True)
    return X_exp / (1 + partition)  # 这里应用了广播机制


class SpaceFeatureAttention(nn.Module):
    def __init__(self, in_channels, out_channels, channels, samples=128, dropout=0.5):
        super(SpaceFeatureAttention, self).__init__()
        self.samples = samples
        self.channels = channels
        self.depth = out_channels
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.depth * self.samples) ** 0.5
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv13 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv14 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.alpha = nn.Parameter(data=torch.zeros(size=(1, self.depth, self.channels, self.samples)),
                                  requires_grad=True)

    def forward(self, x):
        # B 1 C T
        res = x
        # B 4 C T
        out1, out2, out3 = self.conv11(x), self.conv12(x), self.conv13(x)
        query = rearrange(out1, 'b c s t -> b s (c t)')
        key = rearrange(out2, 'b c s t -> b (c t) s')
        value = rearrange(out3, 'b c s t -> b (c t) s')
        attention = torch.einsum('b s t, b t c -> b s c', query, key) / self.scale
        attention_score = self.dropout(softmax(attention))
        out = torch.einsum('b c s, b t s -> b c t', attention_score, value)
        # B C (4 * T) -> B 4 C T
        out = rearrange(out, 'b s (c t) -> b c s t', c=self.depth)
        out = self.conv14(out)
        out = self.alpha * out + res

        return out


class TimeFeatureAttention(nn.Module):
    def __init__(self, in_channels, out_channels, channels, samples=128, dropout=0.5):
        super(TimeFeatureAttention, self).__init__()
        self.samples = samples
        self.channels = channels
        self.depth = out_channels
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.depth * self.channels) ** 0.5
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv13 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv14 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.beta = nn.Parameter(data=torch.zeros(size=(1, 1, self.channels, self.samples)), requires_grad=True)

    def forward(self, x):
        # B 1 C T
        res = x
        # B 4 C T
        out1, out2, out3 = self.conv11(x), self.conv12(x), self.conv13(x)
        # B T (4 * C)
        query = rearrange(out1, 'b c s t -> b t (c s)')
        # B (4 * C) T
        key = rearrange(out2, 'b c s t -> b (c s) t')
        value = rearrange(out3, 'b c s t -> b (c s) t')
        attention = torch.einsum('b s t, b t c -> b s c', query, key) / self.scale
        attention_score = self.dropout(softmax(attention))
        out = torch.einsum('b c s, b t s -> b c t', attention_score, value)
        # B T (4 * C) -> B 4 C T
        out = rearrange(out, 'b s (c t) -> b c t s', c=self.depth)
        out = self.conv14(out)
        out = self.beta * out + res

        return out


class TimeSpaceFeatureExtract(nn.Module):
    def __init__(self, depth=8, channels=24, samples=128, drop_out=0.5):
        super(TimeSpaceFeatureExtract, self).__init__()
        self.channels = channels
        self.samples = samples
        self.depth = depth
        self.drop_out = drop_out
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.depth, kernel_size=(1, 125), bias=False, padding=(0, (125 - 1) // 2)),
            nn.BatchNorm2d(self.depth)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.depth, out_channels=self.depth * 2, kernel_size=(24, 1), bias=False),
            nn.BatchNorm2d(self.depth * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.depth * 2, out_channels=self.depth * 2, kernel_size=(1, 15), bias=False, padding=(0, (15 - 1) // 2)),
            nn.Conv2d(in_channels=self.depth * 2, out_channels=self.depth * 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.depth * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out)
        )

        self.time_attention = TimeFeatureAttention(in_channels=1, out_channels=self.depth, channels=self.channels,
                                                   samples=self.samples)
        self.space_attention = SpaceFeatureAttention(in_channels=self.depth, out_channels=self.depth,
                                                     channels=self.channels,
                                                     samples=self.samples)

    def forward(self, x):
        x = self.time_attention(x)
        x = self.block_1(x)
        x = self.space_attention(x)
        x = self.block_2(x)
        x = self.block_3(x)

        return x


class FrequencyFeatureAttention(nn.Module):
    def __init__(self, in_channels, out_channels, channels, samples, dropout=0.25):
        super(FrequencyFeatureAttention, self).__init__()
        self.channels = channels
        self.samples = samples
        self.depth = out_channels
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.depth, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.depth),
            nn.Conv2d(in_channels=self.depth, out_channels=1, kernel_size=1, stride=1),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.scale = (self.depth * self.samples) ** 0.5
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv13 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv14 = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(data=torch.zeros(size=(1, 1, self.channels, self.samples)), requires_grad=True)

    def forward(self, x):
        x = self.conv3(x)
        # B C C T
        res = x
        # B 4 C T
        out1, out2, out3 = self.conv11(x), self.conv12(x), self.conv13(x)
        # B T (4 * C)
        query = rearrange(out1, 'b c s t -> b s (c t)')
        key = rearrange(out2, 'b c s t -> b (c t) s')
        value = rearrange(out3, 'b c s t -> b (c t) s')
        attention = torch.einsum('b s t, b t c -> b s c', query, key) / self.scale
        attention_score = self.dropout(softmax(attention))
        out = torch.einsum('b c s, b t s -> b c t', attention_score, value)
        # B C (4 * T) -> B 4 C T
        out = rearrange(out, 'b s (c t) -> b c s t', c=self.depth)
        out = self.conv14(out)
        out = self.gamma * out + res

        return out


class FrequencyFeatureExtract(nn.Module):
    def __init__(self, depth=8, channels=24, samples=128, dropout=0.5):
        super(FrequencyFeatureExtract, self).__init__()
        self.depth = depth
        self.channels = channels
        self.samples = samples
        self.dropout = dropout
        self.frequency_attention = FrequencyFeatureAttention(in_channels=self.channels, out_channels=self.depth,
                                                             channels=self.channels, samples=samples)
        self.conv = nn.Sequential(
            # 改了out_channels=8和16

            nn.Conv2d(in_channels=1, out_channels=self.depth, kernel_size=(1, 125), stride=1, bias=False, padding=(0, (125 - 1) // 2)),
            nn.BatchNorm2d(self.depth),

            nn.Conv2d(in_channels=self.depth, out_channels=self.depth * 2, kernel_size=(self.channels, 1), stride=1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),


            nn.Conv2d(in_channels=self.depth * 2, out_channels=self.depth * 2, kernel_size=(1, 15), stride=1, bias=False, padding=(0, (15 - 1) // 2)),
            nn.Conv2d(in_channels=self.depth * 2, out_channels=self.depth * 2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.depth * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.frequency_attention(x)
        x = self.conv(x)

        return x


class FusionAttention(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5, bias=False):
        super(FusionAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(self.in_dim, self.out_dim, bias=bias)
        self.key = nn.Linear(self.in_dim, self.out_dim, bias=bias)
        self.value = nn.Linear(self.in_dim, self.out_dim, bias=bias)
        self.out = nn.Linear(self.out_dim, self.in_dim, bias=bias)
        self.scaler = self.out_dim ** 1/2
        self.norm = nn.LayerNorm(self.in_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim * 4, bias=bias),
            nn.Linear(self.in_dim * 4, self.in_dim, bias=bias)
        )

    def forward(self, out_space_time, out_frequency):
        # shape = B 16 1 4
        if out_space_time.ndim == 4:
            out_space_time = out_space_time.squeeze(2)
            out_frequency = out_frequency.squeeze(2)

        query = self.query(out_space_time)              # shape = B 16 dim
        key = self.key(out_frequency).permute(0, 2, 1)  # shape = B dim 16
        value = self.value(out_frequency)

        attention = torch.bmm(query, key) / self.scaler
        attention_score = self.dropout(softmax(attention))
        outputs = torch.bmm(attention_score, value)               # shape = B 16 dim
        outputs = self.out(outputs)
        outputs = self.norm(outputs)
        outputs = self.feed_forward(outputs)

        return outputs

class FusionMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, d_k, num_heads, dropout=0.5, bias=False):
        super(FusionMultiHeadAttention, self).__init__()
        self.in_dim = in_dim
        self.d_k = d_k
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(self.in_dim, self.d_k, bias=bias)
        self.key = nn.Linear(self.in_dim, self.d_k, bias=bias)
        self.value = nn.Linear(self.in_dim, self.d_k, bias=bias)
        self.out = nn.Linear(self.d_k, self.in_dim, bias=bias)
        self.scaler = self.d_k ** (1/2)


    def forward(self, out_space_time, out_frequency):
        if out_space_time.ndim == 4:
            out_space_time = out_space_time.squeeze(2)
            out_frequency = out_frequency.squeeze(2)

        # B 16 d_k
        queries = rearrange(self.query(out_space_time), 'b n (h d) -> b h n d', h=self.num_heads)
        keys = rearrange(self.key(out_frequency), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.value(out_frequency), 'b n (h d) -> b h n d', h=self.num_heads)

        attention = torch.einsum('b h q d, b h k d -> b h q k', queries, keys) / self.scaler
        attention_score = self.dropout(softmax(attention))

        outputs = torch.einsum('b h a l, b h l v -> b h a v', attention_score, values)

        outputs = rearrange(outputs, 'b h n d -> b n (h d)', h=self.num_heads)
        outputs = self.out(outputs)

        return outputs


class TSFF(nn.Module):
    def __init__(self, num_classes=3, samples=1000, channels=3, depth=8):
        super(TSFF, self).__init__()
        self.channels = channels
        self.samples = samples
        self.depth = depth

        self.sapce_time_extract = TimeSpaceFeatureExtract(depth=self.depth, channels=self.channels,
                                                          samples=self.samples)

        self.frequeny_feature_extract = FrequencyFeatureExtract(depth=self.depth, channels=self.channels,
                                                                samples=self.samples)

        self.cls = nn.Linear(self._get_in_params, num_classes)

        # 方法一：weighted sum
        self.weight_index = nn.Parameter(torch.randn((1,)), requires_grad=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(self._get_in_params, 128),
            nn.ELU(),
            nn.Linear(128, self._get_in_params)
        )

        # 方法二：Attention
        # self.fusion_attention = FusionAttention(in_dim=self._get_in_params[1], out_dim=self.depth * 2)

        # 方法三：MutiAttention
        # self.fusion_muti_attention = FusionMultiHeadAttention(in_dim=self._get_in_params[1], d_k=self._get_in_params[1] * 4, num_heads=4)

    @property
    def _get_in_params(self):
        with torch.no_grad():
            mode = 1
            # 方法一，加权
            if mode == 1:
                out1 = torch.randn((1, 1, 24, 128))
                out1 = self.sapce_time_extract(out1)
                out = out1.view(out1.shape[0], -1)

                return out.shape[-1]
            elif mode == 2:
                out1 = torch.randn((1, 1, 24, 128))
                out2 = torch.randn((1, 24, 24, 128))
                out1 = self.sapce_time_extract(out1)

                return out1.shape[-1] * out1.shape[1], out1.shape[-1]

    def forward(self, x, x_frequency):
        out_space_time = self.sapce_time_extract(x)
        # out_time = self.time_feature_extract(x)
        out_frequency = self.frequeny_feature_extract(x_frequency)

        # 方法一
        out_space_time = out_space_time.view(out_space_time.shape[0], -1)
        out_frequency = out_frequency.view(out_frequency.shape[0], -1)
        out = self.weight_index * out_space_time + (1 - self.weight_index) * out_frequency
        out = self.feed_forward(out)

        # 方法二
        # out = self.fusion_attention(out_space_time, out_frequency)
        # out = out.view(out.shape[0], -1)


        out = self.cls(out)
        return out
