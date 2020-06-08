import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=50, num_layers=3, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
