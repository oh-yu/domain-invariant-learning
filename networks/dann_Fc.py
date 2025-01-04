import torch
from torch import nn, optim

from ..algo import supervised_algo
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder
from .base import SupervisedBase


class Dann_F_C(SupervisedBase):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.encoder = Conv2d().to(self.device)
        self.decoder = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048).to(self.device)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 10