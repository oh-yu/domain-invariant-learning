import torch
import torch.nn.functional as F
from torch import nn


class Conv1dThreeLayers(nn.Module):
    # TODO: Understand nn.Conv1d doumentation
    def __init__(self, input_size: int, out_channels1: int = 128, out_channels2: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels1, kernel_size=8, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels1)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels1, out_channels=out_channels2, kernel_size=5, stride=1, padding=0
        )
        self.bn2 = nn.BatchNorm1d(out_channels2)

        self.conv3 = nn.Conv1d(in_channels=out_channels2, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = torch.mean(x, dim=2)
        return x
