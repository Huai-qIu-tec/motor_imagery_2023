import numpy as np
import os
import pickle

import pywt
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from model.csp import csp
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

mean_std = {}
mean_std_frequency = {}
csp_W = {}
for s in range(1, 10):
    sub = s
    print('sub %d is preprocessing !' % sub)
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
        unpreprcess_data = data['data']
        person = data['personID']
        # 通道名称
        ch_names = data['ch_names']
        # 获取trigger导
        trigger = unpreprcess_data[-1, :]
        trigger_idx = np.where(trigger != 0)[0]
        trigger_content = trigger[trigger_idx]
        # 选择电极通道
        useful_channels = ['FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3',
                           'CP4', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
        ch_idx = [True if i in useful_channels else False for i in ch_names] + [False]
        # ch_idx = [0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 38, 42, 43, 44, 45, 46, 49, 50, 51]
        unpreprcess_data = unpreprcess_data[ch_idx, :]


        def cz_reference(signals, cz_index):
            # 获取Cz通道数据
            cz_data = signals[cz_index]
            # 重参考，逐通道减去Cz通道的值
            num_channels, num_samples = signals.shape
            for i in range(num_channels):
                if i != cz_index:
                    signals[i] -= cz_data

            return signals


        # 1000HZ --> 250HZ 降采样，结果放在raw_data中
        for i in range(1, len(trigger_idx) - 2, 6):
            signal_index = trigger_idx[i: i + 6]
            extract_signals = np.concatenate((unpreprcess_data[:, signal_index[0]:signal_index[1]:4],
                                              unpreprcess_data[:, signal_index[2]:signal_index[3]:4],
                                              unpreprcess_data[:, signal_index[4]:signal_index[5]:4]), axis=1)
            # extract_signals = unpreprcess_data[:, signal_index[0]:signal_index[1]:4]
            # 重参考
            extract_signals = cz_reference(extract_signals, 2)

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


    def __get_sub_wavelet_data(signals):
        def wavelet_transform(signal, scales, wavelet, fs):
            # 初始化时频图矩阵
            t, c, s = signal.shape
            time_freq = np.zeros((t, c, scales.shape[0], s))

            for i in range(t):
                index = 0
                for j in range(c):
                    # 获取当前通道的信号
                    eeg = signal[i, j]
                    # 连续小波变换
                    coeffs, freqs = pywt.cwt(eeg, scales, wavelet, 1. / fs)
                    # 提取时频图
                    time_freq[i, index, :, :] = np.square(coeffs)
                    index += 1

            return time_freq.reshape((t, c, -1, s)), freqs

        # 定义连续小波变换的参数
        wavelet = 'morl'  # 选择小波类型，这里以Morlet小波为例
        fc = pywt.central_frequency(wavelet)
        cparam = 2 * fc * signals.shape[-1]
        scales = cparam / np.arange(28, 4, -1)
        fs = 250
        time_freq, freqs = wavelet_transform(signals, scales, wavelet, fs)
        return time_freq, freqs


    time_freq, freqs = __get_sub_wavelet_data(signals=window_data)
    print(time_freq.shape)

    # 滤波 + 去趋势
    window_data = signal.detrend(window_data)
    window_data = signal.filtfilt(b, a, window_data)


    def norm(data, mode='Standardization'):
        """
            对数据进行归一化
            :param data:   ndarray ,shape[N,channel,samples]
            """
        data_copy = np.copy(data)
        if mode == 'Standardization':
            mean_std['mean_sub%d' % sub] = np.mean(data_copy, axis=(0, 1))
            mean_std['std_sub%d' % sub] = np.std(data_copy, axis=(0, 1))
            data_copy = (data_copy - np.mean(data_copy, axis=(0, 1), keepdims=True)) / (
                    np.std(data_copy, axis=(0, 1), keepdims=True) + 1e-6)
        elif mode == 'MinMaxScaler':
            scaler = MinMaxScaler()
            mean_std['scaler_sub%d' % sub] = scaler
            for i in range(data_copy.shape[0]):
                data_copy[i] = scaler.fit_transform(data_copy[i])

        return data_copy


    def norm2(data, mode='Standardization'):
        """
            对数据进行归一化
            :param data:   ndarray ,shape[N,channel,samples]
            """
        data_copy = np.copy(data)
        if mode == 'Standardization':
            mean_std_frequency['mean_sub%d' % sub] = np.mean(data_copy, axis=(0, 1, 2))
            mean_std_frequency['std_sub%d' % sub] = np.std(data_copy, axis=(0, 1, 2))
            data_copy = (data_copy - np.mean(data_copy, axis=(0, 1, 2), keepdims=True)) / (
                    np.std(data_copy, axis=(0, 1, 2), keepdims=True) + 1e-6)
        elif mode == 'MinMaxScaler':
            scaler = MinMaxScaler()
            mean_std_frequency['scaler_sub%d' % sub] = scaler
            for i in range(data_copy.shape[0]):
                for j in range(data_copy.shape[1]):
                    data_copy[i, j] = scaler.fit_transform(data_copy[i, j])

        return data_copy


    preprocessed_signals = norm(window_data[:index])
    time_freq = norm2(time_freq)
    print(preprocessed_signals.shape)

    # 划分训练集、验证集和测试集下标
    mask1, mask2, mask3 = batch // 3, batch // 3 * 2, batch
    data_sub1, label_sub1 = preprocessed_signals[:mask1], labels[:mask1]
    data_sub2, label_sub2 = preprocessed_signals[mask1:mask2], labels[mask1:mask2]
    data_sub3, label_sub3 = preprocessed_signals[mask2:mask3], labels[mask2:mask3]
    data_frequency_sub1 = time_freq[:mask1]
    data_frequency_sub2 = time_freq[mask1:mask2]
    data_frequency_sub3 = time_freq[mask2:mask3]


    # 数据增强
    def augment(signals, labels):
        signals1 = signals * -1
        labels1 = labels

        def swap(signals1, signals2):
            temp = signals1
            signals1[:] = signals2
            signals2[:] = temp

            return signals1, signals2

        signals2 = np.zeros(signals.shape)
        labels2 = labels
        central = [2, 17]
        for i in range(24):
            if i in central:
                signals2[i] = signals[i]
            if i not in central:
                signals2[i], signals2[i + 1] = swap(signals2[i], signals2[i + 1])

        return np.concatenate((signals1, signals2), axis=0), np.concatenate((labels1, labels2), axis=0)


#     print('begin split augment and norm!')
#     # 划分训练集、验证集和测试集
#     # Block1 Block2 训练集 Block3验证集
#     # 把Block1和Block2拼接
#     train_data_sub1, val_data_sub1 = np.concatenate([data_sub1, data_sub2], axis=0), data_sub3
#     # 把Block1和Block2的时频拼接
#     train_frequencydata_sub1, val_frequencydata_sub1 = np.concatenate([data_frequency_sub1, data_frequency_sub2],
#                                                                       axis=0), data_frequency_sub3
#     # 把Block1和Block2的labels拼接
#     y_train_sub1, y_val_sub1 = np.concatenate([label_sub1, label_sub2], axis=0), label_sub3
#     # 对拼接后的数据进行增强并把增强的数据计算时频
#     aug_train_data_sub1, aug_y_train_sub1 = augment(train_data_sub1, y_train_sub1)
#     aug_train_frequencydata_sub1 = norm2(__get_sub_wavelet_data(aug_train_data_sub1)[0])
#     # 拼接增强后的数据和时频数据和labels
#     train_data_sub1 = np.concatenate((train_data_sub1, aug_train_data_sub1), axis=0)
#     train_frequencydata_sub1 = np.concatenate((train_frequencydata_sub1, aug_train_frequencydata_sub1), axis=0)
#     y_train_sub1 = np.concatenate((y_train_sub1, aug_y_train_sub1), axis=0)
#     print(train_data_sub1.shape, train_frequencydata_sub1.shape)
#     print('block1 done!')
#
#     # Block1 Block3 训练集 Block2验证集
#     train_data_sub2, val_data_sub2 = np.concatenate([data_sub1, data_sub3], axis=0), data_sub2
#     train_frequencydata_sub2, val_frequencydata_sub2 = np.concatenate([data_frequency_sub1, data_frequency_sub3],
#                                                                       axis=0), data_frequency_sub2
#     y_train_sub2, y_val_sub2 = np.concatenate([label_sub1, label_sub3], axis=0), label_sub2
#     aug_train_data_sub2, aug_y_train_sub2 = augment(train_data_sub2, y_train_sub2)
#     aug_train_frequencydata_sub2 = norm2(__get_sub_wavelet_data(aug_train_data_sub2)[0])
#     train_data_sub2 = np.concatenate((train_data_sub2, aug_train_data_sub2), axis=0)
#     train_frequencydata_sub2 = np.concatenate((train_frequencydata_sub2, aug_train_frequencydata_sub2), axis=0)
#     y_train_sub2 = np.concatenate((y_train_sub2, aug_y_train_sub2), axis=0)
#     print('block2 done!')
#
#     # Block2 Block3 训练集 Block1验证集
#     train_data_sub3, val_data_sub3 = np.concatenate([data_sub2, data_sub3], axis=0), data_sub1
#     train_frequencydata_sub3, val_frequencydata_sub3 = np.concatenate([data_frequency_sub2, data_frequency_sub3],
#                                                                       axis=0), data_frequency_sub1
#     y_train_sub3, y_val_sub3 = np.concatenate([label_sub2, label_sub3], axis=0), label_sub1
#     aug_train_data_sub3, aug_y_train_sub3 = augment(train_data_sub3, y_train_sub3)
#     aug_train_frequencydata_sub3 = norm2(__get_sub_wavelet_data(aug_train_data_sub3)[0])
#     train_data_sub3 = np.concatenate((train_data_sub3, aug_train_data_sub3), axis=0)
#     train_frequencydata_sub3 = np.concatenate((train_frequencydata_sub3, aug_train_frequencydata_sub3), axis=0)
#     y_train_sub3 = np.concatenate((y_train_sub3, aug_y_train_sub3), axis=0)
#     print('block3 done!')
#
#     data = {
#         'train_data_sub1': train_data_sub1, 'train_label_sub1': y_train_sub1,
#         'val_data_sub1': val_data_sub1, 'val_label_sub1': y_val_sub1,
#         'train_data_sub2': train_data_sub2, 'train_label_sub2': y_train_sub2,
#         'val_data_sub2': val_data_sub2, 'val_label_sub2': y_val_sub2,
#         'train_data_sub3': train_data_sub3, 'train_label_sub3': y_train_sub3,
#         'val_data_sub3': val_data_sub3, 'val_label_sub3': y_val_sub3,
#     }
#     file = open('../mi/MI/S' + str(sub) + '.pkl', 'wb')
#     pickle.dump(data, file)
#     file.close()
#
#     data_frequency = {
#         'train_frequency_data_sub1': train_frequencydata_sub1, 'val_frequency_data_sub1': val_frequencydata_sub1,
#         'train_frequency_data_sub2': train_frequencydata_sub2, 'val_frequency_data_sub2': val_frequencydata_sub2,
#         'train_frequency_data_sub3': train_frequencydata_sub3, 'val_frequency_data_sub3': val_frequencydata_sub3,
#     }
#     file = open('../mi/MI/S%d_frequency.pkl' % sub, 'wb')
#     pickle.dump(data_frequency, file)
#     file.close()
#     print('sub %d done' % sub)
#
file = open('../mi/MI/mean_std.pkl', 'wb')
pickle.dump(mean_std, file)
file = open('../mi/MI/mean_std_frequency.pkl', 'wb')
pickle.dump(mean_std_frequency, file)
print('%s preprocessing done!' % {'train_data' if train_mode else 'test_data'})