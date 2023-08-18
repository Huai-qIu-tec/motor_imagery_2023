import argparse
import itertools
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from model.SpaceTemporalSpectralMultiAttention import TSFF
from motor_dataset_frequency import MotorImageryFusion
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import confusion_matrix

train_data_dir = '../mi/MI'
test_data_dir = './data/test_data'

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

import warnings
warnings.filterwarnings("ignore")


def get_fusion_dataloader(data_dir, batch_size, split_sub):
    [data_dir, frequency_dir] = data_dir
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)

    with open(frequency_dir, 'rb') as f:
        frequency_data = pickle.load(f)

    train_data, train_label = data['train_data_sub%d' % split_sub], data['train_label_sub%d' % split_sub]
    val_data, val_label = data['val_data_sub%d' % split_sub], data['val_label_sub%d' % split_sub]

    train_frequency_data, val_frequency_data = \
        frequency_data['train_frequency_data_sub%d' % split_sub], frequency_data['val_frequency_data_sub%d' % split_sub]

    train_dataset = MotorImageryFusion(train_data, train_frequency_data, train_label)
    val_dataset = MotorImageryFusion(val_data, val_frequency_data, val_label)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader


def plot_confusion_matrix(net, test_lodaer, sub):
    y_pred = []
    y_true = []
    classes = ['left', 'right', 'foot']
    label = range(3)
    with torch.no_grad():
        for step, data in enumerate(test_lodaer):
            [signals, frequencies], labels = data
            signals, frequencies = signals.cuda(), frequencies.cuda()
            labels = labels.cuda()
            out = net(signals.unsqueeze(1), frequencies)
            predict_labels = torch.argmax(out, dim=-1)
            y_true.append(labels.cpu().detach().numpy())
            y_pred.append(predict_labels.cpu().detach().numpy())

    y_pred = np.array(y_pred).reshape((-1,))
    y_true = np.array(y_true).reshape((-1,))
    cm = confusion_matrix(y_true, y_pred, labels=label, normalize='true')
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('sub %d confusion matrix' % (int(sub)), fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=25)
    plt.yticks(tick_marks, classes, fontsize=25)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=30)

    plt.ylabel('True Label', fontsize=30)
    plt.xlabel('Predict Label', fontsize=30)
    plt.savefig('pic/confusion_matrix.png', dpi=300)
    plt.show()


def set_rng_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return acc, precision, recall, f1


def train_epoch(net, loss, scheduler, optimizer, train_loader):
    running_loss = 0.0
    acc = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    train_num = len(train_loader.dataset)
    net.train()
    for step, data in enumerate(train_loader):
        [signals, frequencies], labels = data
        signals, frequencies, labels = signals.cuda(), frequencies.cuda(), labels.cuda().long()
        y_hat = net(signals.unsqueeze(1), frequencies)
        predict_label = torch.max(y_hat, dim=1)[1]
        a, p, r, f = metric(labels.cpu().detach().numpy(), predict_label.cpu().detach().numpy())
        acc += a
        precision += p
        recall += r
        f1 += f
        l = loss(y_hat, labels)
        optimizer.zero_grad()
        torch.use_deterministic_algorithms(False)
        l.backward()
        optimizer.step()
        scheduler.step()
        running_loss += l.sum().item() * signals.size(0)

    return running_loss / train_num, [acc / (step + 1), precision / (step + 1), recall / (step + 1), f1 / (step + 1)]


def validate(net, test_loader, loss):
    net.eval()
    acc = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    test_num = len(test_loader.dataset)
    running_loss = 0.0
    step = 0
    with torch.no_grad():
        for s, val_data in enumerate(test_loader):
            [val_signals, val_frequencies], val_labels = val_data
            val_signals, val_frequencies, val_labels = val_signals.cuda(), val_frequencies.cuda(), val_labels.long().cuda()
            outputs = net(val_signals.unsqueeze(1), val_frequencies)
            predict_labels = torch.max(outputs, dim=1)[1]
            l = loss(outputs, val_labels)
            a, p, r, f = metric(val_labels.cpu().detach().numpy(), predict_labels.cpu().detach().numpy())
            acc += a
            precision += p
            recall += r
            f1 += f
            running_loss += l.sum().item() * val_signals.size(0)
            step = s

    return running_loss / test_num, [acc / (step + 1), precision / (step + 1), recall / (step + 1), f1 / (step + 1)]


def train(net, loss, optimizer, scheduler, train_loader, val_loader, checkpoint_root, epochs, batch_size,
          sub, block, checkpoint=None):
    test_loader = None
    if checkpoint is None:
        best_acc = 0.0
    else:
        if checkpoint['sub'] != sub:
            best_acc = 0.0
        else:
            best_acc = checkpoint['best_acc']
    train_accurates = []
    valid_accurates = []
    train_loss_ls = []
    valid_loss_ls = []

    for epoch in range(epochs):
        # train
        train_running_loss, train_metric_score = train_epoch(net, loss, scheduler, optimizer, train_loader)
        train_loss_ls.append(train_running_loss)
        train_accurates.append(train_metric_score[0])

        # validate
        # valid_running_loss, test_steps, val_accurate, val_labels, predict_label = validate(net, test_loader, loss)
        valid_running_loss, val_metric_score = validate(net, val_loader, loss)
        valid_accurates.append(val_metric_score[0])
        valid_loss_ls.append(valid_running_loss)

        print(
            '[sub %d][block %d][epoch %d] train loss: %.4f valide loss: %.4f train_accuracy: %.4f val_accuracy: %.4f, val_precision: %.4f,'
            'val_recall: %.4f, val_f1: %.4f, best_val_acc: %.4f'
            % (
            sub, block, epoch + 1, train_running_loss, valid_running_loss, train_metric_score[0], val_metric_score[0],
            val_metric_score[1], val_metric_score[2], val_metric_score[3], best_acc))
        if val_metric_score[0] > best_acc:
            best_acc = val_metric_score[0]
            torch.save(
                {'model_state_dict': net.state_dict(), 'best_acc': best_acc, 'sub': sub}, checkpoint_root)
            print('[sub %d][block %d][epoch %d] save model!' % (sub, block, epoch + 1))

    print('Finishing Training')
    print('best acc: %.3f' % best_acc)

    acc_list = [train_accurates, valid_accurates]
    loss_list = [train_loss_ls, valid_loss_ls]

    return best_acc, acc_list, loss_list, test_loader


def main():
    # 参数设置
    average_acc = []
    average_test_acc = []
    sub_acc_list, sub_loss_list = [], []
    test_lodaer_list = []
    for s in [1, 5, 7, 9, 2, 3, 4, 6, 8]:
        sub = s
        print('----------------sub {} begin training---------------------'.format(sub))
        train_dir = ['./mi/MI/frequency' + '/S' + str(sub) + '.pkl', './mi/MI/frequency' + '/S' + str(sub) + '_frequency.pkl']
        batch_size = 64
        lr = 2e-4
        epochs = 300
        net_name = 'fusion'
        checkpoint_root = 'checkpoints/%s_checkpoints/' % net_name
        log_root = './log'
        log_writer = open(log_root + '/%s.txt' % net_name, 'a')
        set_rng_seed(42)

        loss = nn.CrossEntropyLoss()

        block_acc = []
        for split_block in range(1, 4):
            train_loader, val_loader = get_fusion_dataloader(train_dir, batch_size, split_block)
            net = TSFF(num_classes=3, samples=128, channels=24, depth=8).cuda()
            # 计算参数数量
            total_params = 0
            for name, param in net.named_parameters():
                if param.requires_grad:
                    total_params += param.numel()

            print("Total number of parameters:", total_params)
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            # 使用迁移学习，将sub5的权重初始化
            if sub == 1:
                checkpoint_dir = checkpoint_root + '%s_S%d_block%d' % (net_name, sub, split_block) + '.pth'
            elif sub in [5]:
                checkpoint_dir = checkpoint_root + '%s_S%d_block%d' % (net_name, 1, split_block) + '.pth'
            else:
                checkpoint_dir = checkpoint_root + '%s_S%d_block%d' % (net_name, 5, split_block) + '.pth'
            print('%s 权重装填' % checkpoint_dir)
            checkpoint = None
            if os.path.exists(checkpoint_dir):
                checkpoint = torch.load(checkpoint_dir)
                model_state_dict = checkpoint['model_state_dict']
                net.load_state_dict(model_state_dict)

            # 固定当前权重地址
            checkpoint_dir = checkpoint_root + '%s_S%d_block%d' % (net_name, sub, split_block) + '.pth'
            print('train len: %d val len: %d test len: %d' % (
                len(train_loader.dataset), len(val_loader.dataset), len(val_loader.dataset)))
            best_acc, acc_list, loss_list, test_loader = train(net=net,
                                                                  loss=loss,
                                                                  optimizer=optimizer,
                                                                  scheduler=scheduler,
                                                                  train_loader=train_loader,
                                                                  val_loader=val_loader,
                                                                  checkpoint_root=checkpoint_dir,
                                                                  epochs=epochs,
                                                                  batch_size=batch_size,
                                                                  sub=sub,
                                                                  block=split_block,
                                                                  checkpoint=checkpoint)

            block_acc.append(best_acc)
            sub_acc_list.append(acc_list)
            sub_loss_list.append(loss_list)

            # 测试集
            if os.path.exists(checkpoint_dir):
                checkpoint = torch.load(checkpoint_dir)
                model_state_dict = checkpoint['model_state_dict']
                net.load_state_dict(model_state_dict)

            test_running_loss, test_metric_score = validate(net, val_loader, loss)

            test_print = '[sub %d][block %d]test score: acc: %.4f precision: %.4f recall: %.4f f1: %.4f' % (
                sub, split_block, test_metric_score[0], test_metric_score[1], test_metric_score[2],
                test_metric_score[3])

            print(test_print)
            log_writer.write(test_print + "\n")

            average_acc.append(best_acc)
            average_test_acc.append(test_metric_score[3])
            test_lodaer_list.append(val_loader)

        print('****sub%d acc: %.4f****' % (sub, sum(block_acc) / len(block_acc)))

    # 绘制混淆矩阵
    plot_confusion_matrix(net, test_lodaer_list, checkpoint_root, net_name)

    print('\n')
    index = 0
    for i in range(len(average_acc)):
        print('[sub %d][block %d] val acc: %.4f, test acc: %.4f' % (
        i // 3 + 1, i % 3 + 1, average_acc[i], average_test_acc[i]))
        index += 1
        if index % 3 == 0:
            print('[sub %d] average acc: %.4f' % (sub, sum(average_acc[i - 2:i + 1]) / 3))

    print('9名被试验证集平均acc: %.4f' % (sum(average_acc) / len(average_acc)))
    print('9名被试测试集平均acc: %.4f' % (sum(average_test_acc) / len(average_test_acc)))

    fig, axs = plt.subplots(18, 3, figsize=(9, 36))
    for j in range(0, 18, 2):
        for k in range(3):
            ax1 = axs[j, k]

            ax1.plot(sub_acc_list[j + k][0])
            ax1.plot(sub_acc_list[j + k][1])

            ax1.grid(True)
            ax1.legend(['Train Acc', 'Valid Acc'])
            ax1.set_title('Subject %d Acc' % int(j // 2 + 1))

            ax2 = axs[j + 1, k]
            ax2.plot(sub_loss_list[j + k][0])
            ax2.plot(sub_loss_list[j + k][1])
            ax2.grid(True)
            ax2.legend(['Train Loss', 'Valid Loss'])
            ax2.set_title('Subject %d Loss' % int(j // 2 + 1))

    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()
    plt.savefig('pic/loss and acc.png', dpi=300)

main()
