import numpy as np
import torch
import pickle

from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


def get_dataloader(train_dir, test_dir, batch_size):
    train_dataset = MotorImagery(train_dir, True)
    test_dataset = MotorImagery(test_dir, False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader


def aug_data(timg, label, batch_size):
    timg = timg.numpy()
    label = label.numpy()
    aug_data = []
    aug_label = []
    for cls4aug in range(3):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        shape = timg.shape
        tmp_aug_data = np.zeros((int(batch_size / 10), shape[1], shape[2]))
        for ri in range(int(batch_size / 10)):
            for rj in range(5):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 5)
                tmp_aug_data[ri, :, rj * (shape[2] // 5):(rj + 1) * (shape[2] // 5)] = \
                    tmp_data[rand_idx[rj], :,rj * (shape[2] // 5):(rj + 1) * (shape[2] // 5)]

        aug_data.append(tmp_aug_data)
        aug_label.append(np.concatenate([tmp_label, tmp_label, tmp_label]))
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.tensor(aug_data, dtype=torch.float32)
    aug_label = torch.tensor(aug_label, dtype=torch.long)
    return aug_data, aug_label


def aug_data_time_freqency(signals, labels):
    signals = torch.tensor(signals, dtype=torch.float32)
    alpha = torch.tensor([15], dtype=torch.float32)
    sigma = torch.tensor([0.5], dtype=torch.float32)
    shape = signals.shape
    fs = 200
    new_signals = torch.zeros(size=(shape[0] * 2, shape[1], shape[2]), dtype=torch.float32)
    for i in range(signals.shape[1]):
        results = torch.stft(signals[:, i, :], n_fft=fs, return_complex=True, onesided=True)
        cpx = results.clone()
        A = torch.abs(cpx)
        fai = torch.angle(cpx)
        noise1 = torch.normal(0, sigma.item(), size=A.shape)
        noise2 = torch.normal(0, sigma.item(), size=A.shape)
        A1 = A + alpha * noise1
        A2 = A + alpha * noise2
        cpx1 = A1 * torch.exp(1j * fai)
        cpx2 = A2 * torch.exp(1j * fai)
        results1 = cpx1
        results2 = cpx2
        time1 = torch.istft(results1, fs)
        time2 = torch.istft(results2, fs)
        new_signals[:, i, :] = torch.cat((time1[:signals.shape[-1]], time2[:signals.shape[-1]]), dim=0)
        labels = torch.cat((labels, labels), dim=0)
        shuffle_idx = torch.randperm(new_signals.shape[0])
        new_signals = new_signals[shuffle_idx]
        labels = labels[shuffle_idx]

    return new_signals, labels


class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。


    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
