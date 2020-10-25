from torch.utils import data
import numpy as np
import os
from augmentation import spec_augment


class DatasetCSV(data.Dataset):
    """
    Dataset using CSV files
    """
    def __init__(self, data_root, csv_file, is_train, use_spec_augment=False):
        self.data = []
        self.is_train = is_train
        self.spec_aug = use_spec_augment

        with open(csv_file, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if len(line):
                    feat_file, label, snr = line.split(',')
                    feat = np.load(os.path.join(data_root, feat_file))
                    self.data.append([feat, int(label), int(float(snr))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        mean = np.mean(x)
        x -= mean
        if self.is_train and self.spec_aug:            
            x = spec_augment(x)
        y = self.data[index][1]
        snr = self.data[index][2]
        if x.ndim == 2:
            x = x.reshape((1, x.shape[0], x.shape[1]))

        return x, y, snr


class DatasetPickle(data.Dataset):
    """
    Dataset using pickle files
    """
    def __init__(self, data_pickle, is_train, use_spec_augment=False, distil=False):
        self.data = data_pickle
        self.is_train = is_train
        self.spec_aug = use_spec_augment
        self.distil = distil

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        mean = np.mean(x)
        x -= mean
        if self.is_train and self.spec_aug:
            x = spec_augment(x)
        y = self.data[index][1]
        snr = self.data[index][2]
        if x.ndim == 2:
            x = x.reshape((1, x.shape[0], x.shape[1]))

        if not self.distil:
            return x, y, snr
        else:
            yt = self.data[index][3]
            return x, y, snr, yt
