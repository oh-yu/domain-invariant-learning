from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc1(x))