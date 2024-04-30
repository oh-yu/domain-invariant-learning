import torch
from torch import nn, optim

from ..algo import algo
from .conv1d import Conv1d
from .mlp_decoder_domain import DomainDecoder
from .mlp_decoder_task import TaskDecoder
from .conv2d import Conv2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IsihDanns:
    """
    TODO: Attach paper
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lr_dim1: float,
        lr_dim2: float,
        num_epochs_dim1: int,
        num_epochs_dim2: int,
        output_size: int = 1,
        experiment: str = "HHAR",
        is_target_weights: bool = True,
    ):
        if experiment in ["HHAR", "ECOdataset", "ECOdataset_synthetic"]:
            self.feature_extractor = Conv1d(input_size=input_size).to(DEVICE)
            self.domain_classifier_dim1 = DomainDecoder(input_size=hidden_size, output_size=1).to(DEVICE)
            self.task_classifier_dim1 = TaskDecoder(input_size=hidden_size, output_size=output_size).to(DEVICE)
            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim1)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=lr_dim1)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=lr_dim1)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = num_epochs_dim1

            self.task_classifier_dim2 = TaskDecoder(input_size=hidden_size, output_size=output_size).to(DEVICE)
            self.domain_classifier_dim2 = DomainDecoder(input_size=hidden_size, output_size=1).to(DEVICE)
            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim2)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=lr_dim2)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=lr_dim2)
            self.num_epochs_dim2 = num_epochs_dim2
            self.is_target_weights = is_target_weights


        elif experiment in ["MNIST"]:
            self.feature_extractor = Conv2d().to(DEVICE)
            self.task_classifier_dim1 = DomainDecoder(input_size=1600, output_size=10, fc2_size=50).to(DEVICE)
            self.domain_classifier_dim1 = DomainDecoder(input_size=1600, output_size=1, fc2_size=50).to(DEVICE)
            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim1)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=lr_dim1)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=lr_dim1)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = num_epochs_dim1

            self.task_classifier_dim2 = DomainDecoder(input_size=1600, output_size=10, fc2_size=50).to(DEVICE)
            self.domain_classifier_dim2 = DomainDecoder(input_size=1600, output_size=1, fc2_size=50).to(DEVICE)
            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim2)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=lr_dim2)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=lr_dim2)
            self.num_epochs_dim2 = num_epochs_dim2
            self.is_target_weights = is_target_weights


    def fit_1st_dim(self, source_loader, target_loader, test_target_X: torch.Tensor, test_target_y_task: torch.Tensor):
        self.feature_extractor, self.task_classifier_dim1, _ = algo.fit(
            source_loader,
            target_loader,
            test_target_X,
            test_target_y_task,
            self.feature_extractor,
            self.domain_classifier_dim1,
            self.task_classifier_dim1,
            self.criterion,
            self.feature_optimizer_dim1,
            self.domain_optimizer_dim1,
            self.task_optimizer_dim1,
            num_epochs=self.num_epochs_dim1,
            is_target_weights=self.is_target_weights,
        )

    def fit_2nd_dim(self, source_loader, target_loader, test_target_X: torch.Tensor, test_target_y_task: torch.Tensor):
        self.feature_extractor, self.task_classifier_dim2, _ = algo.fit(
            source_loader,
            target_loader,
            test_target_X,
            test_target_y_task,
            self.feature_extractor,
            self.domain_classifier_dim2,
            self.task_classifier_dim2,
            self.criterion,
            self.feature_optimizer_dim2,
            self.domain_optimizer_dim2,
            self.task_optimizer_dim2,
            num_epochs=self.num_epochs_dim2,
            is_psuedo_weights=True,
            is_target_weights=self.is_target_weights,
        )

    def predict(self, X: torch.Tensor, is_1st_dim: bool) -> torch.Tensor:
        if is_1st_dim:
            return self.task_classifier_dim1.predict(self.feature_extractor(X))
        else:
            return self.task_classifier_dim2.predict(self.feature_extractor(X))

    def predict_proba(self, X: torch.Tensor, is_1st_dim: bool) -> torch.Tensor:
        if is_1st_dim:
            return self.task_classifier_dim1.predict_proba(self.feature_extractor(X))
        else:
            return self.task_classifier_dim2.predict_proba(self.feature_extractor(X))

    def set_eval(self):
        self.task_classifier_dim2.eval()
        self.feature_extractor.eval()
