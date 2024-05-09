import torch
from torch import nn

from .conv1d import Conv1d
from .mlp_decoder_one_layer import OneLayerDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoDATS_F_C(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.conv1d = Conv1d(input_size=input_size).to(DEVICE)
        self.decoder = OneLayerDecoder(input_size=128, output_size=output_size).to(DEVICE)
    def forward(self, x):
        return self.decoder(self.conv1d(x))
    
    def predict(self, x):
        return self.decoder.predict(self.conv1d(x))
    
    def predict_proba(self, x):
        return self.decoder.predict_proba(self.conv1d(x))
