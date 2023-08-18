import numpy as np
import os
import pickle

import torch
from scipy import signal

'''
    define const variable and preprocess signals
        1. load data
        2. band pass filter
        3. standard normalization
'''

train_data_dir = '../data/train_data'
test_data_dir = '../data/test_data'

# 采样率
sample_rate = 250.
# 训练集 or 测试集
train_mode = False
# 带通滤波
Wl, Wh = 4., 40.
Wn = [2 * Wl / sample_rate, 2 * Wh / sample_rate]
[b, a] = signal.butter(N=6, Wn=Wn, btype='bandpass')


for s in range(1, 4):
    sub = s
    loop = range(1, 4) if train_mode else range(4, 7)
    train_str = 'train' if train_mode else 'test'
    window_size = 125
    stride = 125
    channels = 59
    index = 0
    # 批量数
    batch = ((3950 - window_size) // stride + 1) * 10 * 3 * 3
    window_data = np.zeros((batch, channels, window_size), dtype=np.float32)
    labels = np.zeros((batch,), dtype=np.int32)

    for block in loop:
        if train_mode:
            sub_data_dir = os.path.join(train_data_dir, 'S' + str(sub), 'block' + str(block) + '.pkl')
        else:
            sub_data_dir = os.path.join(test_data_dir, 'S' + str(sub), 'block' + str(block) + '.pkl')

        with open(sub_data_dir, 'rb') as f:
            data = pickle.load(f)

        # 脑电数据+trigger
        unpreprcess_data = data['data']
        person = data['personID']
        # 通道名称
        ch_names = data['ch_names']
        # 获取trigger导
        trigger = unpreprcess_data[-1, :]
        trigger_idx = np.where(trigger != 0)[0]
        trigger_content = trigger[trigger_idx]
        # 选择电极通道
        useful_channels = ['FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7']
        ch_idx = [True if i in useful_channels else False for i in ch_names] + [False]
        ch_idx = [0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 38, 42, 43, 44, 45, 46, 49, 50, 51]
        unpreprcess_data = unpreprcess_data[0:59, :]

        # 滤波
        for i in range(channels):
            unpreprcess_data[i, :] = signal.detrend(unpreprcess_data[i, :])
            unpreprcess_data[i, :] = signal.filtfilt(b, a, unpreprcess_data[i, :])

        # 1000HZ --> 250HZ 降采样，结果放在raw_data中

        for i in range(1, len(trigger_idx) - 2, 2):
            signal_index = trigger_idx[i: i + 2]
            extract_signals = unpreprcess_data[:, signal_index[0]:signal_index[1]]
            if trigger[signal_index[0]] in [11.0, 12.0, 13.0]:
                num_windows = (extract_signals.shape[1] - window_size) // stride + 1

                for j in range(num_windows):
                    start_index = j * stride
                    end_index = start_index + window_size
                    if np.sum(extract_signals[:, start_index:end_index]) == 0:
                        continue
                    window_data[index, :, :] = extract_signals[:, start_index:end_index]
                    labels[index] = 0
                    index = index + 1

            elif trigger[signal_index[0]] in [21.0, 22.0, 23.0]:
                num_windows = (extract_signals.shape[1] - window_size) // stride + 1
                for j in range(num_windows):
                    start_index = j * stride
                    end_index = start_index + window_size
                    if np.sum(extract_signals[:, start_index:end_index]) == 0:
                        continue
                    window_data[index, :, :] = extract_signals[:, start_index:end_index]
                    labels[index] = 1
                    index = index + 1

            elif trigger[signal_index[0]] in [31.0, 32.0, 33.0]:
                num_windows = (extract_signals.shape[1] - window_size) // stride + 1
                for j in range(num_windows):
                    start_index = j * stride
                    end_index = start_index + window_size
                    if np.sum(extract_signals[:, start_index:end_index]) == 0:
                        continue
                    window_data[index, :, :] = extract_signals[:, start_index:end_index]
                    labels[index] = 2
                    index = index + 1



    # shuffle
    shuffle_index = np.random.permutation(window_data.shape[0])
    window_data = window_data[shuffle_index]
    labels = labels[shuffle_index]

    def norm(data):
        """
            对数据进行归一化
            :param data:   ndarray ,shape[N,channel,samples]
            """
        data_copy = np.copy(data)
        data_copy = (data_copy - np.mean(data_copy, axis=2, keepdims=True)) / (np.std(data_copy, axis=2, keepdims=True) + 1e-6)
        # for i in range(len(data)):
        #     data_copy[i] = data_copy[i] / (np.max(abs(data[i])) + 1e-6)
        # data_copy = data_copy / (np.max(np.abs(data_copy), axis=(1, 2), keepdims=True) + 1e-6)
        return data_copy

    # standard normalization
    raw_data = norm(window_data[:index])
    print(raw_data.shape)
    data = {'data': raw_data, 'label': labels[:index]}
    file = open('../data/' + train_str + '_data/S' + str(sub) + '_' + train_str + '.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
print('%s preprocessing done!' % {'train_data' if train_mode else 'test_data'})
