from abc import ABC, abstractmethod
from absl import flags
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..algo import coral_algo, dann_algo, jdot_algo
from ..utils import utils
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder


class danns_base(ABC):
    def __init__(self, experiment: str) -> None:
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR", "MNIST"]
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
            self.domain_classifier = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(self.device)
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
        
        elif experiment == "MNIST":
            self.device = torch.device("cpu")

            self.feature_extractor = Conv2d().to(self.device)
            self.task_classifier = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048).to(
                self.device
            )
            self.domain_classifier = ThreeLayersDecoder(input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024).to(
                self.device
            )
            self.domain_criterion = nn.BCELoss()

            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=1e-4)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=1e-4)
            self.num_ecochs = 100

            self.is_target_weights = False
            self.batch_size = 64
