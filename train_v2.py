import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from model.LMDANet import LMDA
from motor_dataset_cross import MotorImagery
from torch.utils.data import DataLoader, random_split
from model.EEGNet import EEGNet
from model.space_time_wise_attention import ViT

train_data_dir = './mi/MI'
test_data_dir = './data/test_data'
import warnings

warnings.filterwarnings("ignore")


def plot_confusion_matrix(net, sub_test_lodaer, checkpoint_root, net_name):
    net.eval()
    cm_list = []
    classes = ['left', 'right', 'foot']
    label = range(3)
    for sub, test_lodaer in enumerate(sub_test_lodaer):
        checkpoint = torch.load(checkpoint_root + '%s_cross_S' % net_name + str(sub + 1) + '.pth')
        model_state_dict = checkpoint['model_state_dict']
        net.load_state_dict(model_state_dict)
        y_pred = []
        y_true = []
        with torch.no_grad():
            for step, (signals, labels) in enumerate(test_lodaer):
                signals = signals.cuda()
                labels = labels.cuda()
                outputs = net(signals.unsqueeze(1))
                predict_labels = torch.max(outputs, dim=1)[1]
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(predict_labels.cpu().detach().numpy())

        y_pred = np.array(y_pred).reshape((-1,))
        y_true = np.array(y_true).reshape((-1,))
        cm = confusion_matrix(y_true, y_pred, labels=label, normalize='true')
        cm_list.append(cm)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # 遍历每个混淆矩阵并绘制
    for i, ax in enumerate(axes.flatten()):
        cm = cm_list[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Subject {i + 1}')
        fig.colorbar(im, ax=ax)

        # 在每个格子上显示相应的值
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, format(cm[j, k], '.2f'), ha='center', va='center', color='white')
        plt.title('sub %d confusion matrix' % (int(sub)))
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes)
        ax.set_yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig('pic/confusion_matrix.png', dpi=300)


def get_dataloader(train_dir, test_dir, batch_size, cross=True):
    if cross:
        train_dataset = MotorImagery(train_dir, train=True, cross=True)
        val_dataset = MotorImagery(test_dir, train=False, cross=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    else:
        train_dataset = MotorImagery(test_dir, train=True, cross=False)
        val_dataset = MotorImagery(test_dir, train=False, cross=False)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return train_dataloader, val_dataloader


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
        signals, labels = data
        signals, labels = signals.cuda(), labels.cuda().long()
        y_hat = net(signals.unsqueeze(1))
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
    with torch.no_grad():
        for step, val_data in enumerate(test_loader):
            val_signals, val_labels = val_data
            val_signals = val_signals.cuda()
            val_labels = val_labels.long().cuda()
            outputs = net(val_signals.unsqueeze(1))
            predict_labels = torch.max(outputs, dim=1)[1]
            l = loss(outputs, val_labels)
            a, p, r, f = metric(val_labels.cpu().detach().numpy(), predict_labels.cpu().detach().numpy())
            acc += a
            precision += p
            recall += r
            f1 += f
            running_loss += l.sum().item() * val_signals.size(0)

    return running_loss / test_num, [acc / (step + 1), precision / (step + 1), recall / (step + 1), f1 / (step + 1)]


def train(net, loss, optimizer, scheduler, train_loader, val_loader, checkpoint_root, epochs, sub,
          checkpoint=None):
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
            '[sub %d][epoch %d] train loss: %.4f valide loss: %.4f train_accuracy: %.4f val_accuracy: %.4f, val_precision: %.4f,'
            'val_recall: %.4f, val_f1: %.4f, best_val_acc: %.4f'
            % (sub, epoch + 1, train_running_loss, valid_running_loss, train_metric_score[0], val_metric_score[0],
               val_metric_score[1], val_metric_score[2], val_metric_score[3], best_acc))
        if val_metric_score[0] > best_acc:
            best_acc = val_metric_score[0]
            torch.save(
                {'model_state_dict': net.state_dict(), 'best_acc': best_acc, 'sub': sub}, checkpoint_root)
            print('[sub %d][epoch %d] save model!' % (sub, epoch + 1))

    print('Finishing Training')
    print('best acc: %.3f' % best_acc)

    acc_list = [train_accurates, valid_accurates]
    loss_list = [train_loss_ls, valid_loss_ls]

    return best_acc, acc_list, loss_list


def main():
    # 参数设置
    average_acc = []
    average_test_acc = []
    sub_acc_list, sub_loss_list = [], []
    test_lodaer_list = []
    for s in [1, 5, 7, 9, 2, 3, 4, 6, 8]:
        net_name = 'eegnet'
        sub = s
        print('----------------sub {} begin training---------------------'.format(sub))

        train_data_dir = ['./mi/MI' + '/S' + str(i) + '.pkl' for i in range(1, 10) if i != sub]
        test_data_dir = ['./mi/MI' + '/S' + str(sub) + '.pkl']
        batch_size = 32
        lr = 3e-4
        epochs = 150
        checkpoint_root = 'checkpoints/%s_checkpoints/' % net_name
        log_root = './log'
        log_writer = open(log_root + '/%s.txt' % net_name, 'a')

        set_rng_seed(42)
        train_loader, val_loader = get_dataloader(train_data_dir, test_data_dir, batch_size, True)
        train_loader2, val_loader2 = get_dataloader(test_data_dir, test_data_dir, batch_size // 2, False)
        net = EEGNet(3).cuda()

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        # 使用迁移学习，将sub5的权重初始化
        if sub == 1:
            checkpoint_dir = checkpoint_root + '%s_cross_S%d' % (net_name, sub) + '.pth'
        elif sub in [2, 3, 4, 6, 7, 8, 9]:
            checkpoint_dir = checkpoint_root + '%s_cross_S%d' % (net_name, 5) + '.pth'
        else:
            checkpoint_dir = checkpoint_root + '%s_cross_S%d' % (net_name, 1) + '.pth'
        print('%s 权重装填' % checkpoint_dir)
        checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['model_state_dict']
            net.load_state_dict(model_state_dict)

        # 固定当前权重地址
        checkpoint_dir = checkpoint_root + '%s_cross_S%d' % (net_name, sub) + '.pth'

        print('train len: %d val len: %d test len: %d' % (
            len(train_loader.dataset), len(val_loader.dataset), len(val_loader.dataset)))
        best_acc, acc_list, loss_list = train(net=net,
                                              loss=loss,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              train_loader=train_loader,
                                              val_loader=val_loader,
                                              checkpoint_root=checkpoint_dir,
                                              epochs=epochs,
                                              sub=sub,
                                              checkpoint=checkpoint)
        sub_acc_list.append(acc_list)
        sub_loss_list.append(loss_list)

        # 测试集
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['model_state_dict']
            net.load_state_dict(model_state_dict)

        test_running_loss, test_metric_score = validate(net, val_loader, loss)

        test_print = 'test score: acc: %.4f precision: %.4f recall: %.4f f1: %.4f' % (
            test_metric_score[0], test_metric_score[1], test_metric_score[2], test_metric_score[3])

        print('[train across subject]  ', test_print)


        # 开始当前sub训练
        lr = 1e-4
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        checkpoint_dir = checkpoint_root + '%s_cross_S' % net_name + str(sub) + '.pth'
        checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['model_state_dict']
            net.load_state_dict(model_state_dict)
        print('train len: %d val len: %d test len: %d' % (
            len(train_loader2.dataset), len(val_loader2.dataset), len(val_loader2.dataset)))
        best_acc, acc_list, loss_list = train(net=net,
                                              loss=loss,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              train_loader=train_loader2,
                                              val_loader=val_loader2,
                                              checkpoint_root=checkpoint_dir,
                                              epochs=epochs * 2,
                                              sub=sub,
                                              checkpoint=checkpoint)

        sub_acc_list[-1][0].extend(acc_list[0])
        sub_acc_list[-1][1].extend(acc_list[1])
        sub_loss_list[-1][0].extend(loss_list[0])
        sub_loss_list[-1][1].extend(loss_list[1])

        # 测试集
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['model_state_dict']
            net.load_state_dict(model_state_dict)

        test_running_loss, test_metric_score = validate(net, val_loader2, loss)

        test_print = 'test score: acc: %.4f precision: %.4f recall: %.4f f1: %.4f' % (
            test_metric_score[0], test_metric_score[1], test_metric_score[2], test_metric_score[3])

        print('[train inner subject]  ', test_print)
        log_writer.write(test_print + "\n")

        average_acc.append(best_acc)
        average_test_acc.append(test_metric_score[3])
        test_lodaer_list.append(val_loader)

    for i in range(len(average_acc)):
        print('sub %d val acc: %.4f, test acc: %.4f' % (int(i + 1), average_acc[i], average_test_acc[i]))

    print('9名被试验证集平均acc: %.4f' % (sum(average_acc) / len(average_acc)))
    print('9名被试测试集平均acc: %.4f' % (sum(average_test_acc) / len(average_test_acc)))

    # 绘制混淆矩阵
    plot_confusion_matrix(net, test_lodaer_list, checkpoint_root, net_name)

    fig, axs = plt.subplots(2, 9, figsize=(36, 8))
    for j in range(9):
        ax1 = axs[0, j]

        ax1.plot(sub_acc_list[j][0])
        ax1.plot(sub_acc_list[j][1])

        ax1.grid(True)
        ax1.legend(['Train Acc', 'Valid Acc'])
        ax1.set_title('Subject %d Acc' % int(j + 1))

        ax2 = axs[1, j]
        ax2.plot(sub_loss_list[j][0])
        ax2.plot(sub_loss_list[j][1])
        ax2.grid(True)
        ax2.legend(['Train Loss', 'Valid Loss'])
        ax2.set_title('Subject %d Loss' % int(j + 1))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout()
    plt.savefig('pic/loss and acc.png', dpi=300)


main()
