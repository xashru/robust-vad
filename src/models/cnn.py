import torch
import torch.nn as nn
import time
from models.dropblock import DropBlock2D


class CNN60(nn.Module):
    """
    CNN model with ~60K trainable parameters. This is the model used in the paper
    """
    def __init__(self):
        super(CNN60, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=(1, 1))
        self.pool5 = nn.MaxPool2d(kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=64*2*2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool5(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN60BN(nn.Module):
    """
    Same as CNN60, with Batch Normalization added
    """
    def __init__(self):
        super(CNN60BN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_features=64*2*2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.bn1(self.activation(self.conv1(x)))
        x = self.pool5(x)
        x = self.bn2(self.activation(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn3(self.activation(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN100(nn.Module):
    """
    A larger CNN model
    """
    def __init__(self):
        super(CNN100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=(1, 1))
        self.pool5 = nn.MaxPool2d(kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=96*2*2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool5(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNDropBlock(nn.Module):
    """
    CNN60 with DropBlock regularization added
    """
    def __init__(self):
        super(CNNDropBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=(1, 1))
        self.pool5 = nn.MaxPool2d(kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.drop_block = DropBlock2D(block_size=4, drop_prob=0.3)
        self.fc1 = nn.Linear(in_features=64 * 2 * 2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x, prob=0):
        x = self.activation(self.conv1(x))
        x = self.pool5(x)
        x = self.drop_block(x, prob)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def test_cnn():
    net = CNN60()
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    count = []
    for p in net.parameters():
        if p.requires_grad:
            count.append(p.numel())
    print('Parameter counts in layers of CNN')
    for i in range(0, len(count), 2):
        print(count[i] + count[i + 1])
    print('param count in CNN:', total_params)

    x = torch.rand((1, 1, 40, 40))
    num_runs = 20
    start_time = time.time()
    for i in range(num_runs):
        y = net(x)
    time_diff = time.time() - start_time
    print('Average time taken: {} ms'.format(1000*time_diff/num_runs))


if __name__ == '__main__':
    test_cnn()
