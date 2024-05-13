import torch
from torch import nn, optim

from ..algo import algo
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder
from .mlp_decoder_one_layer import OneLayerDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Codats:
    """
    CoDATS model https://arxiv.org/abs/2005.10996
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lr: float,
        num_epochs: int,
        experiment: str,
        num_domains: int = 1,
        num_classes: int = 1,
        is_target_weights: bool = True,
    ) -> None:
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.feature_extractor = Conv1dTwoLayers(input_size=input_size).to(DEVICE)
            self.domain_classifier = ThreeLayersDecoder(input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10).to(DEVICE)
            self.task_classifier = ThreeLayersDecoder(input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10).to(DEVICE)

        elif experiment == "HHAR":
            self.feature_extractor = Conv1dThreeLayers(input_size=input_size).to(DEVICE)
            self.domain_classifier = ThreeLayersDecoder(input_size=hidden_size, output_size=num_domains, dropout_ratio=0.3).to(DEVICE)
            self.task_classifier = OneLayerDecoder(input_size=hidden_size, output_size=num_classes).to(DEVICE)

        self.criterion = nn.BCELoss()
        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=lr)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.is_target_weights = is_target_weights

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
            is_target_weights=self.is_target_weights,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict(self.feature_extractor(x))

    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()
