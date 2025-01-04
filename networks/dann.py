import torch
from torch import nn, optim

from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder
from .base import DannsBase

class Dann(DannsBase):
    def __init__(self, experiment="MNIST"):
        if experiment == "MNIST":
            self.device = torch.device("cpu")

            self.feature_extractor = Conv2d().to(self.device)
            self.task_classifier = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048).to(
                self.device
            )
            self.domain_classifier = ThreeLayersDecoder(input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024).to(
                self.device
            )
            self.criterion = nn.BCELoss()

            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=1e-4)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=1e-4)
            self.num_epochs = 100

            self.is_target_weights = False
            self.batch_size = 64
            self.experiment = experiment
            self.do_early_stop = False
