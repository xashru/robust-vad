import argparse
import numpy as np
import random
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from tqdm import tqdm

import models
from data_loader import DataLoaderCSV, DataLoaderPickle

parser = argparse.ArgumentParser(description='VAD Training and Evaluation')
parser.add_argument('--csv_path', default='data/csv_files', type=str)
parser.add_argument('--data-root', default='data/features', type=str, help='features root directory')
parser.add_argument('--use-pickle', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use pickle format data')
parser.add_argument('--model', default='cnn', type=str, help='name of model')
parser.add_argument('--name', default='vad_cnn', type=str, help='name of run')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=15, type=int, help='total epochs to run')
parser.add_argument('--spec-aug', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use spec augment')
parser.add_argument('--dropblock', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use dropblock in CNN')
parser.add_argument('--alpha', default=0.95, type=float, help='Distillation alpha')
parser.add_argument('--T', default=8, type=int, help='Distillation temperature')


args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_file = 'logs.txt'

data_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 0
}

if args.use_pickle:
    data_loader = DataLoaderPickle(args.data_root)
else:
    data_loader = DataLoaderCSV(args.csv_path, args.data_root)
train_loader, val_loader, test_loaders = data_loader.get_data_loader(data_params, args.spec_aug, distil=True)

criterion = nn.CrossEntropyLoss()
criterion_kldiv = nn.KLDivLoss(reduction='batchmean')
max_epochs = args.epoch
model = models.__dict__[args.model]()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)


def test_snr(y, y_pred, y_snr, snr):
    snr_y = np.zeros(shape=(0,), dtype=int)
    snr_pred_y = np.zeros(shape=(0,), dtype=float)
    correct = 0
    for i in range(y_snr.size):
        if y_snr[i] == snr:
            snr_y = np.append(snr_y, y[i])
            snr_pred_y = np.append(snr_pred_y, y_pred[i])
            if (y[i] == 0 and y_pred[i] <= 0.5) or (y[i] == 1 and y_pred[i] > 0.5):
                correct += 1
    auc = roc_auc_score(snr_y, snr_pred_y)
    fpr, tpr, threshold = roc_curve(snr_y, snr_pred_y, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer *= 100.0
    acc = correct / snr_y.size
    return auc, eer, acc


def test(loader):
    model.eval()
    y_np = np.zeros(shape=(0,), dtype=int)
    y_pred_np = np.zeros(shape=(0,), dtype=float)
    y_snr = np.zeros(shape=(0,), dtype=int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, snr in tqdm(loader, desc='test'):
            x, y = x.to(device), y.to(device)
            x = x.float()
            y_pred = model(x)
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            y_pred_score = y_pred[:, 1]
            y_pred_np_batch = y_pred_score.cpu().detach().numpy()
            y_np_batch = y.cpu().detach().numpy()
            y_snr_batch = snr.cpu().detach().numpy()
            y_np = np.concatenate((y_np, y_np_batch))
            y_pred_np = np.concatenate((y_pred_np, y_pred_np_batch))
            y_snr = np.concatenate((y_snr, y_snr_batch))
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()
            total += x.shape[0]
    # calc
    auc = roc_auc_score(y_np, y_pred_np)
    fpr, tpr, threshold = roc_curve(y_np, y_pred_np, pos_label=1)
    eer = brentq(lambda z: 1. - z - interp1d(fpr, tpr)(z), 0., 1.)
    eer *= 100.0
    acc = correct / total
    test_result = {'auc_avg': 100.0 * auc, 'eer_avg': eer, 'acc_avg': 100.0 * acc}
    print('Test AUC: {}, Test EER: {}, Test Acc: {}'.format(100.0*auc, eer, acc))

    # test for different snr levels
    snr_levels = [20, 15, 10, 5, 0, -5, -10]
    for snr in snr_levels:
        auc, eer, acc = test_snr(y_np, y_pred_np, y_snr, snr)
        test_result['auc_' + str(snr)] = auc * 100.0
        test_result['eer_' + str(snr)] = eer
        test_result['acc_' + str(snr)] = acc * 100.0
    log = 'Test=> ' + json.dumps(test_result) + '\n'
    with open(log_file, 'a') as f:
        f.write(log)


def validate(loader):
    model.eval()
    val_loss = 0
    val_iteration = 0
    y_np = np.zeros(shape=(0,), dtype=int)
    y_pred_np = np.zeros(shape=(0,), dtype=float)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc='validation'):
            x, y = x.to(device), y.to(device)
            x = x.float()
            y_pred = model(x)
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
            val_iteration += 1
            y_pred_score = y_pred[:, 1]
            y_pred_np_batch = y_pred_score.cpu().detach().numpy()
            y_np_batch = y.cpu().detach().numpy()
            y_np = np.concatenate((y_np, y_np_batch))
            y_pred_np = np.concatenate((y_pred_np, y_pred_np_batch))
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()
            total += x.shape[0]
    val_loss /= val_iteration
    acc = correct / total
    auc = roc_auc_score(y_np, y_pred_np)
    fpr, tpr, threshold = roc_curve(y_np, y_pred_np, pos_label=1)
    eer = brentq(lambda z: 1. - z - interp1d(fpr, tpr)(z), 0., 1.)
    eer *= 100.0
    log = 'Validation=> Loss: {}, Accuracy: {}, AUC: {}, EER: {}\n'.format(val_loss, acc, auc, eer)
    print('Validation loss: {},  Acc: {}, AUC: {}, EER: {}'.format(val_loss, acc, auc, eer))
    with open(log_file, 'a') as f:
        f.write(log)
    return val_loss, acc


def train():
    for epoch in range(max_epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        train_iteration = 0
        correct = 0
        total = 0
        print('Epoch: ', epoch)
        with open(log_file, 'a') as f:
            f.write('Epoch: {}\n'.format(epoch))
        for x, y, _, yt in tqdm(train_loader, desc='train'):
            x, y, yt = x.to(device), y.to(device), yt.to(device)
            x = x.float()
            if args.dropblock:
                y_pred = model(x, (1 + epoch) / max_epochs * 0.3)
            else:
                y_pred = model(x)

            optimizer.zero_grad()
            loss = (1. - args.alpha) * criterion(y_pred, y) + args.alpha * criterion_kldiv(
                F.log_softmax(y_pred / args.T, dim=1), F.softmax(yt / args.T, dim=1)) * (args.alpha * args.T * args.T)
            train_loss += loss.item()
            train_iteration += 1

            loss.backward()
            optimizer.step()
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()
            total += x.shape[0]
        train_loss /= train_iteration
        acc = correct / total
        log = 'Train=> Loss: {}, Acc: {} \n'.format(train_loss, acc)
        with open(log_file, 'a') as f:
            f.write(log)
        print('Train loss: {}, Train acc: {}'.format(train_loss, acc))

        validate(val_loader)
        torch.save(model.state_dict(), 'weights/' + args.name + '.pt')
        for test_loader in test_loaders:
            validate(test_loader)


def run():
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')
    start_time = time.time()
    train()
    model.load_state_dict(torch.load('weights/' + args.name + '.pt'))
    test(test_loaders[0])
    for loader in test_loaders[1:]:
        validate(loader)
    end_time = time.time()
    log = 'Time taken: {} minutes\n\n\n'.format((end_time-start_time)/60)
    print(log)
    with open(log_file, 'a') as f:
        f.write(log)


if __name__ == '__main__':
    run()
