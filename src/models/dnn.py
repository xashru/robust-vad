import torch.nn as nn


class DNN20(nn.Module):
    def __init__(self):
        super(DNN20, self).__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(in_features=40*20, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = x[:, :, :, 20:].contiguous()
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
