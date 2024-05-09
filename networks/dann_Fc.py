import torch
from torch import nn

from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder


class Dann_F_C(nn.Module):
    def __init__(self, input_size: int=1152, output_size: int = 10, fc1_size: int = 3072, fc2_size: int = 2048, device = torch.device("cpu")):
        super().__init__()
        self.conv2d = Conv2d().to(device)
        self.decoder = ThreeLayersDecoder(input_size=input_size, output_size=output_size, fc1_size=fc1_size, fc2_size=fc2_size).to(device)
        self.device = device
    
    def forward(self, x):
        return self.decoder(self.conv2d(x))
    
    def predict(self, x):
        return self.decoder.predict(self.conv2d(x))
    
    def predict_proba(self, x):
        return self.decoder.predict_proba(self.conv2d(x))