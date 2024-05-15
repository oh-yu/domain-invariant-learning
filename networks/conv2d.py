import torch.nn.functional as F
from torch import nn


class Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # TODO: Understand nn.Conv2d, nn.MaxPool2d DOC
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)

    def forward(self, x):
        # TODO: Understand how this workaround is working
        # https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-64-1-which-is-output-0-of-asstridedbackward0-is-at-version-3-expected-version-2-instead-hint-the-backtrace-further-a/171826/7
        x = F.relu(nn.functional.conv2d(x, self.conv1.weight.clone(), self.conv1.bias, stride=1, padding=1))
        x = self.max_pool1(x)
        x = F.relu(nn.functional.conv2d(x, self.conv2.weight.clone(), self.conv2.bias, stride=1, padding=1))
        x = self.max_pool2(x)
        x = F.relu(nn.functional.conv2d(x, self.conv3.weight.clone(), self.conv3.bias, stride=1, padding=1))
        N = x.shape[0]
        return x.reshape(N, -1)
