import torch
from torch import nn
from torch import optim

import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IsihDanns:
    def __init__(self, input_size, hidden_size, lr, num_epochs_dim1):
        self.feature_extractor = utils.Conv1d(input_size=input_size).to(DEVICE)
        self.domain_classifier_dim1 = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.task_classifier_dim1 = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=lr)
        self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.num_epochs_dim1 = num_epochs_dim1

    def fit_1st_dim():
        pass
    def fit_2nd_dim():
        pass