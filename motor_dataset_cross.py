import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class MotorImagery(Dataset):
    def __init__(self, data_dir, train=True, cross=True):

        self.data = []
        self.label = []
        for path in data_dir:
            with open(path, 'rb') as f:
                files = pickle.load(f)
            if cross:
                if train:
                    tmp_data = torch.tensor(np.concatenate([files['train_data_sub1'], files['val_data_sub1']]),
                                            dtype=torch.float32)
                    tmp_label = torch.tensor(np.concatenate([files['train_label_sub1'], files['val_label_sub1']]),
                                             dtype=torch.long)
                    aug_data1, aug_labels1 = self.augment1(tmp_data, tmp_label)
                    aug_data2, aug_labels2 = self.augment2(tmp_data, tmp_label)
                    tmp_data = torch.cat((tmp_data, aug_data1, aug_data2), dim=0)
                    tmp_label = torch.cat((tmp_label, aug_labels1, aug_labels2), dim=0)
                else:
                    tmp_data = torch.tensor(np.concatenate([files['train_data_sub1'], files['val_data_sub1']]),
                                            dtype=torch.float32)
                    tmp_label = torch.tensor(np.concatenate([files['train_label_sub1'], files['val_label_sub1']]),
                                             dtype=torch.long)
            else:
                if train:
                    tmp_data = torch.tensor(files['train_data_sub1'], dtype=torch.float32)
                    tmp_label = torch.tensor(files['train_label_sub1'], dtype=torch.long)
                    aug_data1, aug_labels1 = self.augment1(tmp_data, tmp_label)
                    aug_data2, aug_labels2 = self.augment2(tmp_data, tmp_label)
                    tmp_data = torch.cat((tmp_data, aug_data1, aug_data2), dim=0)
                    tmp_label = torch.cat((tmp_label, aug_labels1, aug_labels2), dim=0)
                else:
                    tmp_data = torch.tensor(files['val_data_sub1'], dtype=torch.float32)
                    tmp_label = torch.tensor(files['val_label_sub1'], dtype=torch.long)
            self.data.append(tmp_data)
            self.label.append(tmp_label)

        self.data = torch.cat(self.data)
        self.label = torch.cat(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def augment1(self, signals, labels):
        return signals * -1, labels

    def augment2(self, signals, labels):
        def swap(signals1, signals2):
            temp = signals1
            signals1[:] = signals2
            signals2[:] = temp

            return signals1, signals2

        new_signals = torch.zeros(signals.shape)
        central = [0, 7, 16, 25, 42, 49, 56]
        for i in range(59):
            if i in central:
                new_signals[i] = signals[i]
            if i not in central:
                new_signals[i], new_signals[i+1] = swap(new_signals[i], new_signals[i+1])

        return new_signals, labels









    def augment(self, data, labels):
        shape = data.shape
        aug_data = np.zeros((shape[0] // 3, shape[1], shape[2]))
        aug_label = np.ones((shape[0] // 3,))
        for cls4aug in range(3):
            cls_idx = np.where(labels == cls4aug)
            tmp_data = data[cls_idx]
            tmp_label = labels[cls_idx]
            shape = tmp_data.shape
            tmp_aug_data = np.zeros((aug_data.shape[0] // 3, shape[1], shape[2]))
            for ri in range(tmp_aug_data.shape[0]):
                for rj in range(5):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                    tmp_aug_data[ri, :, rj * shape[2] // 5:(rj + 1) * shape[2] // 5] = \
                        tmp_data[rand_idx[rj], :, rj * shape[2] // 5:(rj + 1) * shape[2] // 5]

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
        return torch.tensor(aug_data, dtype=torch.float32), torch.tensor(aug_label, dtype=torch.long)

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


