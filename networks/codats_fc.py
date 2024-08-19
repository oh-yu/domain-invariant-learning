import torch
from torch import nn, optim

from ..utils import utils
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder
from ..algo import supervised_algo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoDATS_F_C(nn.Module):
    def __init__(self, experiment: str):
        super().__init__()
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.conv1d = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.decoder = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.optimizer = optim.Adam(list(self.conv1d.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.criterion = nn.BCELoss()
            self.num_epochs = 300

        elif experiment == "HHAR":
            self.conv1d = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.decoder = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)
            self.optimizer = optim.Adam(list(self.conv1d.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.criterion = nn.CrossEntropyLoss()
            self.num_epochs = 300

    def fit_without_adapt(self, source_loader):
        data = {"loader": source_loader}
        network = {
            "decoder": self.decoder,
            "encoder": self.conv1d,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
        }
        config = {"use_source_loader": True, "num_epochs": self.num_epochs}
        supervised_algo.fit(data, network, **config)

    def fit_on_target(self, target_prime_loader):
        data = {"loader": target_prime_loader}
        network = {
            "decoder": self.decoder,
            "encoder": self.conv1d,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
        }
        config = {"use_source_loader": False, "num_epochs": self.num_epochs}
        supervised_algo.fit(data, network, **config)

    def forward(self, x):
        return self.decoder(self.conv1d(x))

    def predict(self, x):
        return self.decoder.predict(self.conv1d(x))

    def predict_proba(self, x):
        return self.decoder.predict_proba(self.conv1d(x))
