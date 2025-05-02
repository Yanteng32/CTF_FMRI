import time
from itertools import cycle
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataloader import DataSet, load_data
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from config import get_args
#from cnn3d4l import CNN4D_Net4l, CNN4D_Net3l,  CNN4D_Net4l256tp4
from cnn3d4l import CNN4D_Net4l_128tp4, CNN4D_Net3l_64tp4, CNN4D_Net3l_128tp4, CNN4D_Net4l_64tp4


def train_epoch(epoch, ResNet_3D, train_data, fo):
    ResNet_3D.train()
    n_batches = len(train_data)
    learning_rate = '111'
    # 学习率设置
    if learning_rate == '111':
        if epoch<30:
            LEARNING_RATE = 0.0001
        elif epoch<40:
            LEARNING_RATE = 0.00001
        elif epoch<50:
            LEARNING_RATE = 0.000001
        elif epoch<60:
            LEARNING_RATE = 0.000001
        else:
            LEARNING_RATE = 0.000001
    else:
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

    optimizer = torch.optim.Adam([
        {'params': ResNet_3D.parameters(), 'lr': LEARNING_RATE},
    ], lr=0.0001)

    total_loss = 0
    source_correct = 0
    total_label = 0

    for (data, label) in tqdm(train_data, total=n_batches):

        data, label = data.cuda(), label.cuda()
        output, ftx = ResNet_3D(data)
        loss = F.cross_entropy(output, label)

        _, preds = torch.max(output, 1)
        source_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
        total_label += label.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    acc = source_correct / total_label
    mean_loss = total_loss / n_batches
    print(f'Epoch: [{epoch:2d}], '
          f'Loss: {mean_loss:.6f}, ')
    log_str = 'Epoch: '+str(epoch)\
              +' Loss: '+str(mean_loss)\
              +' train_acc: '+str(acc)+'\n'
    fo.write(log_str)
    del acc, train_data, n_batches
    gc.collect()

    return mean_loss, source_correct, total_label


def val_model(epoch, val_data, ResNet_3D, log_best):
    ResNet_3D.eval()
    print('Test a model on the val data...')
    correct = 0
    total = 0
    total_loss = 0
    true_label = []
    data_pre = []
    with torch.no_grad():
        for images, labels in tqdm(val_data):
            images = images.cuda()
            labels = labels.cuda()
            output, ftx = ResNet_3D(images)
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    mean_loss = total_loss / len(val_data)
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    log_str = 'Epoch: ' + str(epoch) \
              + '\n' \
              + 'TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN: ' + str(FN) \
              + '  ACC:  ' + str(ACC) \
              + '  SEN:  ' + str(SEN) \
              + '  SPE:  ' + str(SPE) \
              + '  AUC:  ' + str(AUC) \
              + '\n'
    log_best.write(log_str)
    del correct, total, true_label, data_pre, images, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss  #, feature_for_GCN


def test_model(test_data, ResNet_3D):
    ResNet_3D.eval()
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    with torch.no_grad():
        for images, labels in tqdm(test_data):
            images = images.cuda()
            labels = labels.cuda()
            output, ftx = ResNet_3D(images)

            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('The result of test data: \n')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    del correct, total, true_label, data_pre, images, labels, output, TN, FP, FN, TP
    gc.collect()
    # return ACC, SEN, SPE, AUC


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_data = load_data(args, args.train_root_path, args.AD_dir, args.CN_dir)
    val_data = load_data(args, args.val_root_path, args.AD_dir, args.CN_dir)
    test_data = load_data(args, args.test_root_path, args.AD_dir, args.CN_dir)

    ResNet_3D = CNN4D_Net4l_64tp4().cuda()

    train_best_loss = 10000
    val_best_loss = 10000
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0
    t_precision = 0
    t_f1 = 0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    test_acc_all = []
    count = 0

    since = time.time()

    for epoch in range(1, args.nepoch + 1):
        # 训练模型
        fo = open("test_log_best_rest12_d14.txt", "a")
        log_best = open('log_best_rest12_d14.txt', 'a')
        train_loss, train_correct, len_train = train_epoch(epoch, ResNet_3D, train_data, fo)
        # 保存模型
        print('model saved...')
        torch.save(ResNet_3D.state_dict(), 'model4d/4DCNN4l64_31_tp4_fmri_epoch_'+str(epoch)+'_an_l430_2_dp0_ln2_rest_12b_d14_b12'+'.pt')
        # 打印当前训练损失、最小损失和准确率
        if train_loss < train_best_loss:
            train_best_loss = train_loss
        train_acc = 100. * train_correct / len_train
        if train_acc > train_best_acc:
            max_acc = train_acc
        print('current loss: ', train_loss, 'the best loss: ', train_best_loss)
        print(f'train_correct/train_data: {train_correct}/{len_train} accuracy: {train_acc:.2f}%')

        # 模型用于验证集并将评价指标写入txt
        #ACC, SEN, SPE, AUC, val_loss, feature_for_GCN = val_model(epoch, val_data, ResNet_3D, log_best)
        ACC, SEN, SPE, AUC, val_loss = val_model(epoch, val_data, ResNet_3D, log_best)
        if ACC > val_best_acc:  # if val_loss < val_best_loss   特征保存条件
            target_best_acc = ACC  # val_best_loss = val_loss
            val_best_acc = target_best_acc
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC
            #np.savez('feature_' + str(count), feature_for_GCN.detach().cpu().numpy())
            #count = count+1

        log_best.write('The best result:\n')
        log_best.write('ACC:  ' + str(val_best_acc) + '  SEN:  ' + str(t_SEN) + '  SPE:  ' + str(
            t_SPE) + '  AUC:  ' + str(t_AUC) + '\n\n')

        print(f'The train acc of this epoch: {train_acc:.2f}%')
        print(f'The best acc: {val_best_acc:.2f}% \n')
        fo.write('train_acc: '+str(train_acc)+' The current total loss: '+str(train_loss)+' The best loss: '+str(train_best_loss)+'\n\n')

        # 保存训练结果用于画图
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        test_acc_all.append(ACC)

        del train_loss, train_correct, len_train, train_acc
        gc.collect()
        fo.close()
        log_best.close()

    # 将模型用于测试集
    test_model(test_data, ResNet_3D)

    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "test_acc_all": test_acc_all}
    )
    train_process.to_csv('train_process_result_rest12_d14.csv')

    # 画图
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.plot(train_process.epoch, train_process.test_acc_all, label="Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig("accuracy_loss_rest12_d14.png")
    #plt.show()


