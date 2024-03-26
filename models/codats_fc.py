import torch
from torch import nn

from ..utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoDATS_F_C(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.conv1d = utils.Conv1d(input_size=input_size).to(DEVICE)
        self.decoder = utils.Decoder(input_size=128, output_size=1).to(DEVICE)
    def forward(self, x):
        return self.decoder(self.conv1d(x))