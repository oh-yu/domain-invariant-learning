import torch
from torch import nn, optim

from .base import SupervisedBase
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoDATS_F_C(SupervisedBase):
    def __init__(self, experiment: str):
        super().__init__()
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.encoder = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.decoder = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.criterion = nn.BCELoss()
            self.num_epochs = 300

        elif experiment == "HHAR":
            self.encoder = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.decoder = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)
            self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.criterion = nn.CrossEntropyLoss()
            self.num_epochs = 300
