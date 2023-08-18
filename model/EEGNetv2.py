# 导入工具包
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


class EEGNet(nn.Module):
    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.5
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=25,  # num_filters
                kernel_size=(1, 5),  # filter size
                bias=False,
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(25),  # output shape (8, C, T)
            nn.Conv2d(
                in_channels=25,  # input shape (8, C, T)
                out_channels=25,  # num_filters
                kernel_size=(59, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(25),  # output shape (16, 1, T)
            nn.ELU(),
            nn.MaxPool2d((1, 3)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,  # input shape (1, C, T)
                out_channels=50,  # num_filters
                kernel_size=(1, 5),  # filter size
                bias=False,
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(25),  # output shape (8, C, T)
            nn.ELU(),
            nn.MaxPool2d((1, 2)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=50,  # input shape (16, 1, T//4)
                out_channels=100,  # num_filters
                kernel_size=(1, 5),  # filter size
                bias=False,
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(100),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.MaxPool2d((1, 2)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,  # input shape (16, 1, T//4)
                out_channels=200,  # num_filters
                kernel_size=(1, 5),  # filter size
                bias=False,
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(200),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.MaxPool2d((1, 2)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )


        self.out = nn.Linear((16 * 4), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class Recognize:
    def __init__(self, net, filterA, filterB):
        self.net = net
        self.filterA = filterA
        self.filterB = filterB

    def predict(self, data, personID=1):
        # print(data.shape)
        shape = data.shape[1]
        window_size = 125
        stride = 25
        channels = 59
        num_windows = (250 - window_size) // stride + 1
        left_eeg_data = np.zeros((num_windows, channels, window_size))
        index = 0
        root_dir = os.path.dirname(os.path.abspath(__file__))
        path = root_dir + '/checkpoints/eegnetv2_S{}.pth'.format(personID)
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']
        self.net.load_state_dict(model_state_dict)
        self.net.cuda()
        self.net.eval()
        # <=2s时
        if shape <= 250:

            tmp_data = signal.filtfilt(self.filterB, self.filterA, data)
            for j in range(num_windows):
                start_index = j * stride
                end_index = start_index + window_size
                if np.sum(data[:, start_index:end_index]) == 0:
                    continue
                left_eeg_data[index, :, :] = tmp_data[:, start_index:end_index]
                index = index + 1

            left_eeg_data = left_eeg_data[:index, :, :]
            norm_data = left_eeg_data / (np.max(np.abs(left_eeg_data), axis=(1, 2), keepdims=True) + 1e-6)
            outputs = self.net(torch.tensor(norm_data, dtype=torch.float32).unsqueeze(1).cuda())
            result = torch.argmax(outputs, dim=1)[0]
            mode, _ = torch.mode(result)
            return mode.item() + 1

        # 3s时
        if 250 < shape <= 375:
            eeg_data = data[:, 250:]
            filter_data = signal.filtfilt(self.filterB, self.filterA, eeg_data)
            filter_data = filter_data / (np.max(np.abs(filter_data), axis=(1, 2), keepdims=True) + 1e-6)
            outputs = self.net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda())
            result = torch.argmax(outputs, dim=1)[0]
            return result.item() + 1

        # 4s时
        if 375 < shape <= 500:
            eeg_data = data[:, 375:]
            filter_data = signal.filtfilt(self.filterB, self.filterA, eeg_data)
            filter_data = filter_data / (np.max(np.abs(filter_data), axis=(1, 2), keepdims=True) + 1e-6)
            outputs = self.net(torch.tensor(filter_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda())
            result = torch.argmax(outputs, dim=1)[0]
            return result.item() + 1
