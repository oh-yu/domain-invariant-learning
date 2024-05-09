import torch
from torch import nn

from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers

from .mlp_decoder_one_layer import OneLayerDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoDATS_F_C(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, experiment:str="ecodataset"):
        super().__init__()
        if experiment in ["ecodataset", "ecodataset_synthetic"]:
            self.conv1d = Conv1dTwoLayers(input_size=input_size).to(DEVICE)
        elif experiment == "HHAR":
            self.conv1d = Conv1dThreeLayers(input_size=input_size).to(DEVICE)
        self.decoder = OneLayerDecoder(input_size=128, output_size=output_size).to(DEVICE)
    def forward(self, x):
        return self.decoder(self.conv1d(x))
    
    def predict(self, x):
        return self.decoder.predict(self.conv1d(x))
    
    def predict_proba(self, x):
        return self.decoder.predict_proba(self.conv1d(x))
