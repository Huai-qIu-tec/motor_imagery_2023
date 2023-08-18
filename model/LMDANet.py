import os
import numpy as np
from scipy import signal
import torch
import torch.nn as nn


class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        """
        :arg
        """
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.transpose(-2, -3)

        # print('查看参数是否变化:', conv.bias)

        return y * self.C * x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, chans=20, samples=1000, num_classes=3, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            # nn.AvgPool2d(kernel_size=(1, avepool)),
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls


class Recognize:
    def __init__(self, net, filterA, filterB):
        self.net = net
        self.filterA = filterA
        self.filterB = filterB

    def predict(self, data, personID=1):
        shape = data.shape[1]
        channels = 59
        root_dir = os.path.dirname(os.path.abspath(__file__))
        path = root_dir + '/checkpoints/lmda_S{}.pth'.format(personID)
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']
        self.net.load_state_dict(model_state_dict)
        self.net.eval()
        # <=2s时
        if shape <= 500:
            eeg_data = data[:, 500:]
            for i in range(channels):
                data[i, :] = signal.detrend(data[i, :])
                data[i, :] = signal.filtfilt(self.filterB, self.filterA, data[i, :])

            filter_data = eeg_data / (np.max(abs(eeg_data)) + 1e-6)
            outputs = self.net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(1))
            result = torch.argmax(outputs, dim=1)
            mode, _ = torch.mode(result)
            return mode.item() + 1

        # 3s时
        if 500 < shape <= 750:
            eeg_data = data[:, 500:]
            for i in range(channels):
                eeg_data[i, :] = signal.detrend(eeg_data[i, :])
                eeg_data[i, :] = signal.filtfilt(self.filterB, self.filterA, eeg_data[i, :])

            filter_data = eeg_data / (np.max(np.abs(eeg_data)) + 1e-6)
            outputs = self.net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
            result = torch.argmax(outputs, dim=1)[0]
            return result.item() + 1
        # 4s时
        if 750 < shape <= 1000:
            eeg_data = data[:, 750:]
            for i in range(channels):
                eeg_data[i, :] = signal.detrend(eeg_data[i, :])
                eeg_data[i, :] = signal.filtfilt(self.filterB, self.filterA, eeg_data[i, :])
            filter_data = eeg_data / (np.max(np.abs(eeg_data)) + 1e-6)
            outputs = self.net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
            result = torch.argmax(outputs, dim=1)[0]
            return result.item() + 1
