import torch
from torch import nn
from torch import optim

from ..utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IsihDanns:
    def __init__(self, input_size, hidden_size, lr_dim1, lr_dim2, num_epochs_dim1, num_epochs_dim2):
        self.feature_extractor = utils.Conv1d(input_size=input_size).to(DEVICE)
        self.domain_classifier_dim1 = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.task_classifier_dim1 = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim1)
        self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=lr_dim1)
        self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=lr_dim1)
        self.criterion = nn.BCELoss()
        self.num_epochs_dim1 = num_epochs_dim1

        self.domain_classifier_dim2 = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.task_classifier_dim2 = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim2)
        self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=lr_dim2)
        self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=lr_dim2)
        self.num_epochs_dim2 = num_epochs_dim2

    def fit_1st_dim(self, source_loader, target_loader, test_target_X, test_target_y_task):
        self.feature_extractor, self.task_classifier_dim1, _ = utils.fit(
            source_loader, target_loader, test_target_X, test_target_y_task,
            self.feature_extractor, self.domain_classifier_dim1, self.task_classifier_dim1, self.criterion,
            self.feature_optimizer_dim1, self.domain_optimizer_dim1, self.task_optimizer_dim1, num_epochs=self.num_epochs_dim1
        )

    def fit_2nd_dim(self, source_loader, target_loader, test_target_X, test_target_y_task):
        self.feature_extractor, self.task_classifier_dim2, _ = utils.fit(
            source_loader, target_loader, test_target_X, test_target_y_task,
            self.feature_extractor, self.domain_classifier_dim2, self.task_classifier_dim2, self.criterion,
            self.feature_optimizer_dim2, self.domain_optimizer_dim2, self.task_optimizer_dim2, num_epochs=self.num_epochs_dim2, is_psuedo_weights=True
        )

    def predict(self, X, is_1st_dim: bool) -> torch.Tensor:
        if is_1st_dim:
            out = self.task_classifier_dim1(self.feature_extractor(X))
            out = torch.sigmoid(out).reshape(-1)
            return out
        else:
            out = self.task_classifier_dim2(self.feature_extractor(X))
            out = torch.sigmoid(out).reshape(-1)
            return out
