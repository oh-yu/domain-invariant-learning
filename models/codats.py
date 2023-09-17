import torch
from torch import nn
from torch import optim

from ..utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Codats:
    def __init__(self, input_size, hidden_size, lr, num_epochs):
        self.feature_extractor = utils.Conv1d(input_size=input_size).to(DEVICE)
        self.domain_classifier = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.task_classifier = utils.Decoder(input_size=hidden_size, output_size=1).to(DEVICE)
        self.criterion = nn.BCELoss()

        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=lr)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def fit(self, source_loader, target_loader, test_target_X, test_target_y_task):
        self.feature_extractor, self.task_classifier, _ = utils.fit(
            source_loader, target_loader, test_target_X, test_target_y_task,
            self.feature_extractor, self.domain_classifier, self.task_classifier, self.criterion,
            self.feature_optimizer, self.domain_optimizer, self.task_optimizer,
            num_epochs=self.num_epochs, is_timeseries=False
        )

    def predict(self, x):
        pred_y_task = self.task_classifier(self.feature_extractor(x))
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        return pred_y_task