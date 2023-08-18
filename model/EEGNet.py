# 导入工具包
import torch
import torch.nn as nn

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CBAM(nn.Module):

    def __init__(self, channel, ratio=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EEGNet(nn.Module):
    def __init__(self, classes_num, drop_out=0.5):
        super(EEGNet, self).__init__()
        self.drop_out = drop_out
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),
                bias=False,
                padding=(0, 64 // 2)
            ),
            nn.BatchNorm2d(8),
            SELayer(channel=8, reduction=4),
            # CBAM(channel=8, ratio=4, kernel_size=3),
        )

        # self.se1 = SELayer(channel=8, reduction=4)

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(24, 1),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),

            # CBAM(channel=16, ratio=4, kernel_size=3),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out),
            # SELayer(channel=16, reduction=4),
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 22),  # filter size
                bias=False,
                padding=(0, 22 // 2)
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False,
            ),  # output shape (16, 1, T//4)

            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out),
            SELayer(channel=16, reduction=8),
            # CBAM(channel=16, ratio=4, kernel_size=3),
        )

        self.out = nn.Sequential(
            nn.Linear(self._get_in_params, self._get_in_params * 2),
            nn.ReLU(),
            nn.Linear(self._get_in_params * 2, classes_num)
            # nn.Linear(self._get_in_params, classes_num)
        )

    @property
    def _get_in_params(self):
        with torch.no_grad():
            out = torch.randn((1, 1, 24, 250))
            out = self.block_3(self.block_2(self.block_1(out)))
        return out.shape[1] * out.shape[-1]

    def forward(self, x):
        x = self.block_1(x)
        # x = self.se1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        # x = self.se2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x