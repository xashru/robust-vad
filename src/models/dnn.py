import torch.nn as nn
import time
import torch


class DNN20(nn.Module):
    """
    DNN with 20 temporal frames input. this is the reference DNN model used in the paper
    """
    def __init__(self):
        super(DNN20, self).__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_features=40*20, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = x[:, :, :, 20:].contiguous()
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


class DNN40(nn.Module):
    """
    DNN with 40 temporal frames input. this is the reference DNN model used in the paper
    """
    def __init__(self):
        super(DNN40, self).__init__()
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(in_features=40*40, out_features=36)
        self.fc2 = nn.Linear(in_features=36, out_features=36)
        self.fc3 = nn.Linear(in_features=36, out_features=36)
        self.fc4 = nn.Linear(in_features=36, out_features=2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


def test_dnn():
    net = DNN40()
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
    test_dnn()
