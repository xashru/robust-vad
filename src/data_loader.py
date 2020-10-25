from torch.utils import data
from dataset import DatasetCSV, DatasetPickle
import os
import pickle


class DataLoaderCSV:
    """
    DataLoaders using CSV files
    """
    def __init__(self, csv_path, data_root):
        self.csv_path = csv_path
        self.data_root = data_root

    def get_data_loader(self, params, spec_aug):
        training_set = DatasetCSV(self.data_root, os.path.join(self.csv_path, 'train.csv'), True, spec_aug)
        val_set = DatasetCSV(self.data_root, os.path.join(self.csv_path, 'validation.csv'), False, spec_aug)
        test_set = DatasetCSV(self.data_root, os.path.join(self.csv_path, 'test.csv'), False, spec_aug)

        train_loader = data.DataLoader(training_set, **params)
        val_loader = data.DataLoader(val_set, **params)
        test_loader = data.DataLoader(test_set, **params)

        return train_loader, val_loader, test_loader


class DataLoaderPickle:
    """
    DataLoaders using pickle files
    """
    def __init__(self, data_root):
        self.data_root = data_root

    def get_data_loader(self, params, spec_aug, distil=False):
        with open(os.path.join(self.data_root, 'train.pickle'), 'rb') as handle:
            training_set = DatasetPickle(pickle.load(handle), True, spec_aug, distil)
        with open(os.path.join(self.data_root, 'validation.pickle'), 'rb') as handle:
            val_set = DatasetPickle(pickle.load(handle), False, spec_aug)
        with open(os.path.join(self.data_root, 'test.pickle'), 'rb') as handle:
            test_set = DatasetPickle(pickle.load(handle), False, spec_aug)

        train_loader = data.DataLoader(training_set, **params)
        val_loader = data.DataLoader(val_set, **params)
        test_loader = data.DataLoader(test_set, **params)

        return train_loader, val_loader, test_loader
