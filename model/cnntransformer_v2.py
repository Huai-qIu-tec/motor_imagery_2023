# -*- coding: UTF-8 -*-

import math
import os

import numpy as np
from scipy import signal
from torch.nn import functional as F
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    function: 位置编码，这里的位置编码用的可学习的变量，没有用transformer的sin和cos
    input: X
    output: X + pos_embedding
    """

    def __init__(self, channels, embedding_size, dropout):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, embedding_size))
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, X):
        out = X + self.pos_embedding  # .to(X.device)
        if self.dropout:
            out = self.dropout(out)

        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, dropout, expansion_rate=4):
        super(PositionWiseFFN, self).__init__()
        self.dropout = dropout
        if self.dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc1 = nn.Linear(ffn_num_input, ffn_num_hiddens * expansion_rate)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ffn_num_hiddens * expansion_rate, ffn_num_outputs)

    def forward(self, X):
        out = self.activation(self.fc1(X))
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        return out


def transpose_qkv(X, num_heads):
    """
    input:      X.shape = (batch_size，查询或者“键－值”对的个数，num_hiddens)
    output:     X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    """
    # 输⼊X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状: (batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状: (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    input:      X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    output:     X.shape = (batch_size，查询或者“键－值”对的个数，num_hiddens)
    """
    # 输入X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

    # 输入X.shape = (batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 输入X.shape = (batch_size, 查询或者“键－值”对的个数, num_heads, num_hiddens/num_heads)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention_weights = None
        self.num_heads = num_heads
        self.attention = self.DotProductAttention
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens)
        self.W_k = nn.Linear(key_size, num_hiddens)
        self.W_v = nn.Linear(value_size, num_hiddens)
        self.W_o = nn.Linear(num_hiddens, num_hiddens)

    def DotProductAttention(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def forward(self, X):
        queries = transpose_qkv(self.W_q(X), self.num_heads)
        keys = transpose_qkv(self.W_k(X), self.num_heads)
        values = transpose_qkv(self.W_v(X), self.num_heads)

        # output.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, ffn_num_input,
                 ffn_num_hiddens):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attention = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_input, dropout)
        self.norm2 = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        # layer1 MultiHeadAttention + LayerNorm + Residual #
        residual = X
        out = self.norm1(X)
        out = self.attention(out)
        out += residual

        # layer2 PositionalWiseNet + LayerNorm + Residual + LayerNorm + Residual #
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out += residual
        return out


class TransformerBlock(nn.Module):
    def __init__(self, channels, query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                 ffn_num_input, ffn_num_hiddens, num_layers):
        super(TransformerBlock, self).__init__()
        self.attention_weights = None
        self.num_heads = num_heads
        self.pos_embedding = PositionalEmbedding(channels, num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i), EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                                               ffn_num_input, ffn_num_hiddens))

    def forward(self, X):
        out = self.pos_embedding(X)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            out = blk(out)
            self.attention_weights[i] = blk.attention.attention_weights
        return out


class ConvBlock(nn.Module):
    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 30), stride=(1, 2), padding=(0, 14),
                      padding_mode='replicate'),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(59, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 5))
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
        )

        self.projection_block = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, X):  # X.shape = (batch_size, 1, channels, samplepoints)
        out = self.shallownet(X)

        out = self.projection_block(out).squeeze(2)

        return out.permute(0, 2, 1)


class ClassificationHead(nn.Module):
    def __init__(self, num_class):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(800, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_class)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class EEGCNNTransformer_v2(nn.Module):
    def __init__(self,
                 channels,
                 query_size=40,
                 key_size=40,
                 value_size=40,
                 num_hiddens=40,
                 num_heads=4,
                 dropout=0.5,
                 ffn_num_input=40,
                 ffn_num_hiddens=40,
                 num_layers=2,
                 out_channels=40,
                 num_class=3):
        super(EEGCNNTransformer_v2, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros((1, 1, num_hiddens)), requires_grad=True)
        self.channels = channels
        self.conv_layer = ConvBlock(out_channels=out_channels)
        self.transformer = TransformerBlock(self.channels, query_size, key_size, value_size, num_hiddens, num_heads,
                                            dropout, ffn_num_input, ffn_num_hiddens, num_layers)
        self.classification = ClassificationHead(num_class)

    def forward(self, X):  # X.shape = (batch_size, channels, sampleponits)

        out = self.conv_layer(X.unsqueeze(1))

        out = self.transformer(out)

        logits = self.classification(out)

        return logits

    def recognize(self, net, data, personID=1):

        # print(data.shape)
        shape = data.shape[1]
        eeg_data = np.zeros((20, 250))
        # <=2s时
        if shape <= 500:
            eeg_data[:, :shape] = data[:, 0:shape:2]
        # 3s时
        if 500 < shape <= 750:
            eeg_data[:, :shape] = data[:, 500: shape]
        # 4s时
        if 750 < shape <= 1000:
            eeg_data[:, :shape] = data[:, 750:: shape]

        filter_data = signal.filtfilt(self.filterB, self.filterA, eeg_data)
        filter_data = (filter_data - np.mean(filter_data, axis=1, keepdims=True)) \
                      / (np.std(filter_data, axis=1, keepdims=True) + 1e-8)

        root_dir = os.path.dirname(os.path.abspath(__file__))
        path = root_dir + '/checkpoints/transformer_S{}.pth'.format(personID)
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']
        net.load_state_dict(model_state_dict)
        net.eval()
        outputs = net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
        result = torch.argmax(outputs, dim=1)[0]

        return result.item() + 1
