from torch.utils import data
from dataset import Dataset
import os


class DataLoader:
    def __init__(self, csv_path, data_root):
        self.csv_path = csv_path
        self.data_root = data_root

    def get_data_loader(self, params, spec_aug):
        training_set = Dataset(self.data_root, os.path.join(self.csv_path, 'train.csv'), True, spec_aug)
        train_loader = data.DataLoader(training_set, **params)

        val_set = Dataset(self.data_root, os.path.join(self.csv_path, 'validation.csv'), False, spec_aug)
        val_loader = data.DataLoader(val_set, **params)

        test_set = Dataset(self.data_root, os.path.join(self.csv_path, 'test.csv'), False, spec_aug)
        test_loader = data.DataLoader(test_set, **params)

        return train_loader, val_loader, test_loader
