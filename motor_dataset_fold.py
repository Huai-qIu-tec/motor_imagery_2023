import numpy as np
import torch
import torcheeg.transforms
from scipy import signal
from torch.utils.data import Dataset


class MotorImagery(Dataset):
    def __init__(self, data, label, train=True):

        if train:
            aug_data1, aug_labels1 = self.augment1(data, label)
            aug_data2, aug_labels2 = self.augment2(data, label)
            aug_data3, aug_labels3 = self.augment2(data, label)
            self.data = np.concatenate((data, aug_data1, aug_data2, aug_data3), axis=0)
            self.label = np.concatenate((label, aug_labels1, aug_labels2, aug_labels3), axis=0)
            # self.data = torch.cat((self.data, aug_data1), dim=0)
            # self.label = torch.cat((self.label, aug_labels1), dim=0)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def augment1(self, signals, labels):
        return signals * -1, labels

    def augment2(self, signals, labels):
        shape = signals.shape
        aug_data = np.zeros((shape[0], shape[1], shape[2]))
        aug_label = np.zeros((shape[0],))
        for cls4aug in range(3):
            cls_idx = np.where(labels == cls4aug)
            tmp_data = signals[cls_idx]
            shape = tmp_data.shape
            tmp_aug_data = np.zeros((aug_data.shape[0] // 3, shape[1], shape[2]))
            for ri in range(tmp_aug_data.shape[0]):
                for rj in range(5):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                    tmp_aug_data[ri, :, rj * (shape[2] // 5):(rj + 1) * (shape[2] // 5)] = \
                        tmp_data[rand_idx[rj], :, (rj * shape[2]) // 5:(rj + 1) * (shape[2] // 5)]

            if cls4aug == 0:
                aug_data[:tmp_aug_data.shape[0], :, :] = tmp_aug_data

            if cls4aug == 1:
                aug_label[tmp_aug_data.shape[0]:tmp_aug_data.shape[0] * 2] = 1
                aug_data[tmp_aug_data.shape[0]:tmp_aug_data.shape[0] * 2, :, :] = tmp_aug_data

            if cls4aug == 2:
                aug_label[tmp_aug_data.shape[0] * 2:tmp_aug_data.shape[0] * 3] = 2
                aug_data[tmp_aug_data.shape[0] * 2:tmp_aug_data.shape[0] * 3, :, :] = tmp_aug_data

        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]
        return aug_data, aug_label

    def augment3(self, signals, labels, alpha=1, sigma=0.05):
        shape = signals.shape
        fs = 250
        new_signals = np.zeros(shape=shape)
        for i in range(signals.shape[0]):
            for j in range(signals.shape[1]):
                f, t, Zxx = signal.stft(signals[i][j], fs=fs, nperseg=250)
                A = np.abs(Zxx)
                fai = np.angle(Zxx)
                noise = np.random.normal(0, sigma, size=A.shape)
                A = A + alpha * noise
                Zxx = A * np.exp(1j * fai)
                time = signal.istft(Zxx, fs)
                new_signals[i, j, :] = time[1][:250]

        return new_signals, labels

    def augment4(self, signals, labels):

        eeg_transform = torcheeg.transforms.RandomNoise(mean=0.0, std=0.5)
        new_signals = eeg_transform(eeg=signals)['eeg']

        return new_signals, labels

import mne
import matplotlib.pyplot as plt
signals = torch.randn((32, 24, 250))
eeg_transform = torcheeg.transforms.RandomNoise(mean=0.0, std=0.5, p=1.)
new_signals = eeg_transform(eeg=signals)['eeg']
print(new_signals.shape)
plt.plot(signals[0][0])
plt.plot(new_signals[0][0])
plt.legend(['raw data', 'noise data'])
plt.show()
print(new_signals.shape)
# ch_names = ['FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
# info = mne.create_info(ch_names=ch_names, sfreq=250)
# raw = mne.io.RawArray(raw_data, info)
# raw.plot()
