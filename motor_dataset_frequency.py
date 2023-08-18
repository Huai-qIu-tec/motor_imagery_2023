import os

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import pickle


class MotorImageryFusion(Dataset):
    def __init__(self, data, data_frequency, label):

        self.data = torch.tensor(data, dtype=torch.float32)
        self.data_frequency = torch.tensor(data_frequency, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return [self.data[idx], self.data_frequency[idx]], self.label[idx]

    def augment(self, data, labels):
        shape = data.shape
        aug_data = np.zeros((shape[0], shape[1], shape[2]))
        aug_label = np.zeros((shape[0],))
        for cls4aug in range(3):
            cls_idx = np.where(labels == cls4aug)
            tmp_data = data[cls_idx]
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


    def aug_data_time_freqency(self, signals, labels):
        signals = torch.tensor(signals, dtype=torch.float32)
        alpha = torch.tensor([1], dtype=torch.float32)
        sigma = torch.tensor([0.15], dtype=torch.float32)
        shape = signals.shape
        fs = 200
        new_signals = torch.zeros(size=(shape[0], shape[1], shape[2]), dtype=torch.float32)
        for i in range(signals.shape[1]):
            results = torch.stft(signals[:, i, :], n_fft=fs, return_complex=True, onesided=True)
            cpx = results.clone()
            A = torch.abs(cpx)
            fai = torch.angle(cpx)
            noise1 = torch.normal(0, sigma.item(), size=A.shape)
            A1 = A + alpha * noise1
            cpx1 = A1 * torch.exp(1j * fai)
            results1 = cpx1
            time1 = torch.istft(results1, fs)
            new_signals[:, i, :] = time1[:, :signals.shape[-1]]
            labels = torch.cat((labels, labels), dim=0)
            shuffle_idx = torch.randperm(new_signals.shape[0])
            new_signals = new_signals[shuffle_idx]
            labels = labels[shuffle_idx]

        return new_signals, labels.long()

    def augment3(self, signals, labels, alpha=1, sigma=0.05):
        alpha = alpha
        sigma = sigma
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


