import torch
from networks.mlp_decoder_domain import DomainDecoder
from rnn import ManyToOneRNN
from torch import nn, optim

from ..algo import algo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rdann:
    """
    R-DANN model https://browse.arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lr: float,
        num_epochs: int,
        num_layers: int = 3,
        num_domains: int = 1,
        num_classes: int = 1,
    ) -> None:
        self.feature_extractor = ManyToOneRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(
            DEVICE
        )
        self.domain_classifier = DomainDecoder(input_size=hidden_size, output_size=num_domains).to(DEVICE)
        self.task_classifier = DomainDecoder(input_size=hidden_size, output_size=num_classes).to(DEVICE)
        self.criterion = nn.BCELoss()

        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=lr)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def fit(
        self,
        source_loader: torch.utils.data.dataloader.DataLoader,
        target_loader: torch.utils.data.dataloader.DataLoader,
        test_target_X: torch.Tensor,
        test_target_y_task: torch.Tensor,
    ) -> None:
        self.feature_extractor, self.task_classifier, _ = algo.fit(
            source_loader,
            target_loader,
            test_target_X,
            test_target_y_task,
            self.feature_extractor,
            self.domain_classifier,
            self.task_classifier,
            self.criterion,
            self.feature_optimizer,
            self.domain_optimizer,
            self.task_optimizer,
            num_epochs=self.num_epochs,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pred_y_task = self.task_classifier(self.feature_extractor(x))
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        return pred_y_task
