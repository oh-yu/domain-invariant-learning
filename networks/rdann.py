import torch
from rnn import ManyToOneRNN
from torch import nn, optim

from networks.mlp_decoder_three_layers import ThreeLayersDecoder
from ..algo import dann_algo
from .base import DannsBase
from ..utils import utils

class Rdann(DannsBase):
    """
    R-DANN model https://browse.arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(
        self, experiment: str
    ) -> None:
        if experiment == "HHAR":
            self.feature_extractor = ManyToOneRNN(input_size=6, hidden_size=128, num_layers=3).to(
                utils.DEVICE
            )
            self.domain_classifier = ThreeLayersDecoder(input_size=128, output_size=1).to(utils.DEVICE)
            self.task_classifier = ThreeLayersDecoder(input_size=128, output_size=10).to(utils.DEVICE)
            self.criterion = nn.BCELoss()

            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-3)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=1e-3)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=1e-3)
            self.num_epochs = 100
