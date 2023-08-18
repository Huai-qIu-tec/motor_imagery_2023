import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import aug_data, get_dataloader, aug_data_time_freqency, LabelSmoothingLoss
from model.model import EEGCNNTransformer
from model.EEGNet import EEGNet
from model.s3t import ViT
from model.cnntransformer_v2 import EEGCNNTransformer_v2
from model.s3t_v2 import ViT_v2
from model.conformer import Conformer
from model.LMDANet import LMDA
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

training_mode = True
train_data_dir = './data/train_data'
test_data_dir = './data/test_data'


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
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')

    return acc, precision, recall, f1


def train_epoch(net, loss, scheduler, optimizer, train_loader, batch_size):
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
        # aug_signals, aug_labels = aug_data(signals, labels, batch_size)
        # aug_signals2, aug_labels2 = aug_data_time_freqency(signals, labels)
        # signals, labels = torch.cat((signals, aug_signals), dim=0).cuda(), torch.cat((labels.long(), aug_labels), dim=0).cuda()
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
        # scheduler.step()
        running_loss += l.sum().item() * signals.size(0)

    return running_loss / train_num, [acc / (step + 1), precision / (step + 1), recall / (step + 1), f1 / (step + 1)]


def validate(net, test_loader, loss, batch_size):
    net.eval()
    acc = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    test_num = len(test_loader.dataset)
    running_loss = 0.0

    # val_labels_ls = torch.Tensor().cuda()
    # predict_labels_ls = torch.Tensor().cuda()
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
            # val_labels_ls = torch.cat([val_labels_ls, val_labels])
            # predict_labels_ls = torch.cat([predict_labels_ls, predict_labels])
            running_loss += l.sum().item() * val_signals.size(0)

    return running_loss / test_num, [acc / (step + 1), precision / (step + 1), recall / (step + 1), f1 / (step + 1)]


def train(net, loss, optimizer, scheduler, train_loader, test_loader, checkpoint_root, epochs, batch_size, log_write,
          checkpoint=None):
    if checkpoint is None:
        best_f1 = 0.0
    else:
        best_f1 = checkpoint['best_f1']
    Y_true = 0.0
    Y_predict = 0.0
    train_accurates = []
    valid_accurates = []
    train_loss_ls = []
    valid_loss_ls = []

    for epoch in range(epochs):
        # train
        train_running_loss, train_metric_score = train_epoch(net, loss, scheduler, optimizer, train_loader, batch_size)
        train_loss_ls.append(train_running_loss)
        train_accurates.append(train_metric_score[0])

        # validate
        # valid_running_loss, test_steps, val_accurate, val_labels, predict_label = validate(net, test_loader, loss)
        valid_running_loss, val_metric_score = validate(net, test_loader, loss, batch_size)
        valid_accurates.append(val_metric_score[0])
        valid_loss_ls.append(valid_running_loss)

        print(
            '[epoch %d] train loss: %.4f valide loss: %.4f train_accuracy: %.4f  val_accuracy: %.4f, val_precision: %.4f,'
            'val_recall: %.4f, val_f1: %.4f, lr: %.5f, best_val_f1: %.4f'
            % (epoch + 1, train_running_loss, valid_running_loss, train_metric_score[0], val_metric_score[0],
               val_metric_score[1],
               val_metric_score[2], val_metric_score[3], optimizer.state_dict()['param_groups'][0]['lr'], best_f1))
        if val_metric_score[-1] > best_f1:
            best_f1 = val_metric_score[-1]
            torch.save(
                {'model_state_dict': net.state_dict(),
                 'best_f1': best_f1},
                checkpoint_root)
        log_write.write(str(epoch) + "\t" + str(val_metric_score[0]) + "\n")

    print('Finishing Training')
    print('best f1: %.3f' % best_f1)
    plt.plot(valid_loss_ls)
    plt.title('loss')
    plt.show()
    plt.plot(valid_accurates)
    plt.title('accuracy')
    plt.show()
    log_write.write('The average accuracy is: ' + str(np.mean(valid_accurates)) + "\n")
    log_write.write('The best accuracy is: ' + str(best_f1) + "\n")

    return best_f1, Y_true, Y_predict


def main():
    # 参数设置
    for s in range(1, 2):
        sub = s

        train_data_dir = './data/train_data' + '/S' + str(sub) + '_train.pkl'
        test_data_dir = './data/test_data' + '/S' + str(sub) + '_test.pkl'
        batch_size = 180
        lr = 5e-5
        epochs = 300
        checkpoint_root = './checkpoints/'
        log_root = './log'
        log_writer = open(log_root + '/S' + str(sub) + '.txt', 'w')

        set_rng_seed(1)

        train_loader, test_loader = get_dataloader(train_data_dir, test_data_dir, batch_size)
        transformer = EEGCNNTransformer(channels=20)
        S3T = ViT()
        eegnet = EEGNet(3)
        transformer_v2 = EEGCNNTransformer_v2(channels=20)
        s3t_v2 = ViT_v2()

        lmda = LMDA(chans=59, samples=250, num_classes=3, depth=2, kernel=75, channel_depth1=6, channel_depth2=3)

        conformer = Conformer(depth=2, num_heads=5, emb_size=20)
        net = conformer.cuda()

        net_name = net.__class__.__name__
        loss = LabelSmoothingLoss(smoothing=0.2)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * 2, eta_min=1e-4)
        if net_name == 'EEGNet':
            checkpoint_dir = checkpoint_root + 'EEGNet_S' + str(sub) + '.pth'
        elif net_name == 'EEGCNNTransformer':
            checkpoint_dir = checkpoint_root + 'transformer_S' + str(sub) + '.pth'
        elif net_name == 'ViT':
            checkpoint_dir = checkpoint_root + 'ViT_S' + str(sub) + '.pth'
        elif net_name == 'EEGCNNTransformer_v2':
            checkpoint_dir = checkpoint_root + 'transformer_v2_S' + str(sub) + '.pth'
        elif net_name == 'ViT_v2':
            checkpoint_dir = checkpoint_root + 'ViT_v2_S' + str(sub) + '.pth'
        elif net_name == 'Conformer':
            checkpoint_dir = checkpoint_root + 'conformer_S' + str(sub) + '.pth'
        elif net_name == 'LMDA':
            checkpoint_dir = checkpoint_root + 'lmda_S' + str(sub) + '.pth'
        else:
            checkpoint_dir = ''

        checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['model_state_dict']
            net.load_state_dict(model_state_dict)

        train(net=net,
              loss=loss,
              optimizer=optimizer,
              scheduler=scheduler,
              train_loader=train_loader,
              test_loader=test_loader,
              checkpoint_root=checkpoint_dir,
              epochs=epochs,
              batch_size=batch_size,
              log_write=log_writer,
              checkpoint=checkpoint)


main()
