from torch import nn, optim

from ..utils import utils
from .base import DannsBase
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder


class Codats(DannsBase):
    """
    CoDATS model https://arxiv.org/abs/2005.10996
    """

    def __init__(self, experiment: str) -> None:
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.device = utils.DEVICE
            self.feature_extractor = Conv1dTwoLayers(input_size=3).to(self.device)
            self.domain_classifier = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(self.device)
            self.task_classifier = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(self.device)

            self.criterion = nn.BCELoss()
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.num_epochs = 300
            self.is_target_weights = True
            self.experiment = experiment
            self.batch_size = 32
            self.do_early_stop = False

        elif experiment == "HHAR":
            self.device = utils.DEVICE
            self.feature_extractor = Conv1dThreeLayers(input_size=6).to(self.device)
            self.domain_classifier = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                self.device
            )
            self.task_classifier = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(self.device)

            self.criterion = nn.BCELoss()
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.num_epochs = 300
            self.is_target_weights = True
            self.experiment = experiment
            self.batch_size = 128
            self.do_early_stop = False
