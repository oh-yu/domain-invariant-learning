from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, fc1_size=50, fc2_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x