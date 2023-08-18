import numpy as np
import os
import pickle

import pywt
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from model.csp import csp
import mne
from mne.viz import plot_raw
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

'''
    define const variable and preprocess signals
        1. load data
        2. band pass filter
        3. standard normalization
'''

train_data_dir = '../mi/MI'
test_data_dir = './data/test_data'

# 采样率
sample_rate = 250.
# 训练集 or 测试集
train_mode = True
# 带通滤波
Wl, Wh = 8., 26.
Wn = [2 * Wl / sample_rate, 2 * Wh / sample_rate]
[b, a] = signal.butter(N=5, Wn=Wn, btype='bandpass')


# mne参数
info = mne.create_info(
            ch_names=["Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
                      "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
                      "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
                      "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
                      "TP7", "TP8", 'Pz', "P3", "P4", "P5", "P6", "P7", "P8", "POz",
                      "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2"],  # "Pz", reference
            ch_types="eeg",  # channel type
            sfreq=250,  # frequency
        )

mean_std = {}
csp_W = {}
for s in range(1, 10):
    sub = s
    loop = range(1, 4) if train_mode else range(4, 7)
    train_str = 'train' if train_mode else 'test'
    window_size = 250
    stride = 125
    channels = 24
    index = 0
    # 批量数
    # batch = (((2050 // 4 - window_size) // stride + 1) + 2 * ((950 // 4 - window_size) // stride + 1)) * 10 * 3 * 3
    batch = ((3950 // 4 - window_size) // stride + 1) * 10 * 3 * 3
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
        unpreprocess_data = data['data']
        person = data['personID']
        # 通道名称
        ch_names = data['ch_names']
        # 获取trigger导
        trigger = unpreprocess_data[-1, :]
        trigger_idx = np.where(trigger != 0)[0]
        trigger_content = trigger[trigger_idx]
        # 选择电极通道
        useful_channels = ['FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3',
                           'CP4', 'CP5', 'CP6',
                           'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
        ch_idx = [True if i in useful_channels else False for i in ch_names] + [False]
        # ch_idx = [0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 38, 42, 43, 44, 45, 46, 49, 50, 51]
        unpreprocess_data = unpreprocess_data[ch_idx, :]

        # 1000HZ --> 250HZ 降采样，结果放在raw_data中
        for i in range(1, len(trigger_idx) - 2, 6):
            signal_index = trigger_idx[i: i + 6]
            extract_signals = np.concatenate((unpreprocess_data[:, signal_index[0]:signal_index[1]:4],
                                              unpreprocess_data[:, signal_index[2]:signal_index[3]:4],
                                              unpreprocess_data[:, signal_index[4]:signal_index[5]:4]), axis=1)
            # extract_signals = unpreprocess_data[:, signal_index[0]:signal_index[1]:4]

            def cz_reference(signals, cz_index):
                # 获取Cz通道数据
                cz_data = signals[cz_index]
                # 重参考，逐通道减去Cz通道的值
                num_channels, num_samples = signals.shape
                for i in range(num_channels):
                    if i != cz_index:
                        signals[i] -= cz_data

                return signals


            def avg_reference(signals, cz_index):
                # 获取全脑平均脑电
                avg_data = np.mean(signals, axis=(0, 1), keepdims=True)
                signals = signals - avg_data

                return signals
            extract_signals = cz_reference(extract_signals, 2)

            if trigger[signal_index[0]] in [11.0, 12.0, 13.0]:
                num_windows = (extract_signals.shape[1] - window_size) // stride + 1

                for j in range(num_windows):
                    start_index = j * stride
                    end_index = start_index + window_size
                    window_data[index, :, :] = extract_signals[:, start_index:end_index]
                    labels[index] = 0
                    index = index + 1

            elif trigger[signal_index[0]] in [21.0, 22.0, 23.0]:
                num_windows = (extract_signals.shape[1] - window_size) // stride + 1
                for j in range(num_windows):
                    start_index = j * stride
                    end_index = start_index + window_size
                    window_data[index, :, :] = extract_signals[:, start_index:end_index]
                    labels[index] = 1
                    index = index + 1

            elif trigger[signal_index[0]] in [31.0, 32.0, 33.0]:
                num_windows = (extract_signals.shape[1] - window_size) // stride + 1
                for j in range(num_windows):
                    start_index = j * stride
                    end_index = start_index + window_size
                    window_data[index, :, :] = extract_signals[:, start_index:end_index]
                    labels[index] = 2
                    index = index + 1

    def avg_reference(signals):
        # 获取全脑平均脑电
        avg_data = np.mean(signals, axis=(0, 1), keepdims=True)
        signals = signals - avg_data

        return signals


    def cz_reference(signals, cz_index):
        # 获取Cz通道数据
        cz_data = signals[cz_index]
        # 重参考，逐通道减去Cz通道的值
        num_channels, num_samples = signals.shape
        for i in range(num_channels):
            if i != cz_index:
                signals[i] -= cz_data

        return signals

    # 重参考
    # for i in range(window_data.shape[0]):
    #     window_data[i] = avg_reference(window_data[i])

    # 滤波 + 去趋势
    window_data = signal.detrend(window_data)
    window_data = signal.filtfilt(b, a, window_data)

    def norm(data, sub):
        """
            对数据进行归一化
            :param data:   ndarray ,shape[N,channel,samples]
            """
        data_copy = np.copy(data)
        mean_std['mean_sub%d' % sub] = np.mean(data_copy, axis=(0, 1))
        mean_std['std_sub%d' % sub] = np.std(data_copy, axis=(0, 1))
        data_copy = (data_copy - np.mean(data_copy, axis=(0, 1), keepdims=True)) / (
                np.std(data_copy, axis=(0, 1), keepdims=True) + 1e-6)
        # data_copy = data_copy / np.max(np.abs(data_copy), axis=2, keepdims=True)
        # scalar = MinMaxScaler()
        # for i in range(len(data_copy)):
        #     data_copy[i] = scalar.fit_transform(data_copy[i])
        # mean_std['scalar%d' % sub] = scalar

        return data_copy

    preprocessed_signals = norm(window_data[:index], sub)

    print(preprocessed_signals.shape)

    mask1, mask2, mask3 = batch // 3, batch // 3 * 2, batch
    data_sub1, label_sub1 = preprocessed_signals[:mask1], labels[:mask1]
    data_sub2, label_sub2 = preprocessed_signals[mask1:mask2], labels[mask1:mask2]
    data_sub3, label_sub3 = preprocessed_signals[mask2:mask3], labels[mask2:mask3]

    # 划分训练集、验证集和测试集
    # Block1 Block2 训练集 Block3验证集
    train_data_sub1, val_data_sub1 = np.concatenate([data_sub1, data_sub2], axis=0), data_sub3
    y_train_sub1, y_val_sub1 = np.concatenate([label_sub1, label_sub2], axis=0), label_sub3

    # Block1 Block3 训练集 Block2验证集
    train_data_sub2, val_data_sub2 = np.concatenate([data_sub1, data_sub3], axis=0), data_sub2
    y_train_sub2, y_val_sub2 = np.concatenate([label_sub1, label_sub3], axis=0), label_sub2

    # Block2 Block3 训练集 Block1验证集
    train_data_sub3, val_data_sub3 = np.concatenate([data_sub2, data_sub3], axis=0), data_sub1
    y_train_sub3, y_val_sub3 = np.concatenate([label_sub2, label_sub3], axis=0), label_sub1

    data = {
        'train_data_sub1': train_data_sub1, 'train_label_sub1': y_train_sub1,
        'val_data_sub1': val_data_sub1, 'val_label_sub1': y_val_sub1,
        'train_data_sub2': train_data_sub2, 'train_label_sub2': y_train_sub2,
        'val_data_sub2': val_data_sub2, 'val_label_sub2': y_val_sub2,
        'train_data_sub3': train_data_sub3, 'train_label_sub3': y_train_sub3,
        'val_data_sub3': val_data_sub3, 'val_label_sub3': y_val_sub3,
    }
    file = open('../mi/MI/S' + str(sub) + '.pkl', 'wb')
    pickle.dump(data, file)
    file.close()

file = open('../mi/MI/mean_std.pkl', 'wb')
pickle.dump(mean_std, file)
# file = open('../mi/MI/csp_W.pkl', 'wb')
# pickle.dump(csp_W, file)
print('%s preprocessing done!' % {'train_data' if train_mode else 'test_data'})