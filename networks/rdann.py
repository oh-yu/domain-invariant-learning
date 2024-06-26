import torch
from rnn import ManyToOneRNN
from torch import nn, optim

from networks.mlp_decoder_three_layers import ThreeLayersDecoder

from ..algo import dann_algo

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
        self.domain_classifier = ThreeLayersDecoder(input_size=hidden_size, output_size=num_domains).to(DEVICE)
        self.task_classifier = ThreeLayersDecoder(input_size=hidden_size, output_size=num_classes).to(DEVICE)
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
        data = {
            "source_loader": source_loader,
            "target_loader": target_loader,
            "target_X": test_target_X,
            "target_y_task": test_target_y_task,
        }
        network = {
            "feature_extractor": self.feature_extractor,
            "domain_classifier": self.domain_classifier,
            "task_classifier": self.task_classifier,
            "criterion": self.criterion,
            "feature_optimizer": self.feature_optimizer,
            "domain_optimizer": self.domain_optimizer,
            "task_optimizer": self.task_optimizer,
        }
        config = {
            "num_epochs": self.num_epochs,
        }
        self.feature_extractor, self.task_classifier, _ = dann_algo.fit(data, network, **config)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pred_y_task = self.task_classifier(self.feature_extractor(x))
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        return pred_y_task
