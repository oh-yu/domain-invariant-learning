import torch
from torch import nn, optim

from ..algo import algo
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder
from .mlp_decoder_one_layer import OneLayerDecoder
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
        experiment: str,
        output_size: int = 1,
        is_target_weights: bool = True,
    ):
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR", "MNIST"]

        if experiment in ["HHAR", "ECOdataset", "ECOdataset_synthetic"]:
            if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
                self.feature_extractor = Conv1dTwoLayers(input_size=input_size).to(DEVICE)
            elif experiment == "HHAR":
                self.feature_extractor = Conv1dThreeLayers(input_size=input_size).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(input_size=hidden_size, output_size=1, dropout_ratio=0.3).to(DEVICE)
            self.task_classifier_dim1 = OneLayerDecoder(input_size=hidden_size, output_size=output_size).to(DEVICE)
            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim1)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=lr_dim1)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=lr_dim1)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = num_epochs_dim1

            
            self.task_classifier_dim2 = OneLayerDecoder(input_size=hidden_size, output_size=output_size).to(DEVICE)
            self.domain_classifier_dim2 = ThreeLayersDecoder(input_size=hidden_size, output_size=1, dropout_ratio=0.3).to(DEVICE)
            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim2)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=lr_dim2)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=lr_dim2)
            self.num_epochs_dim2 = num_epochs_dim2
            self.is_target_weights = is_target_weights

            self.device = DEVICE
            self.stop_during_epochs=False
            


        elif experiment in ["MNIST"]:
            self.feature_extractor = Conv2d()
            self.task_classifier_dim1 = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048)
            self.domain_classifier_dim1 = ThreeLayersDecoder(input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024)
            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=lr_dim1)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=lr_dim1)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=lr_dim1)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = num_epochs_dim1


            self.task_classifier_dim2 = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048)
            self.domain_classifier_dim2 = ThreeLayersDecoder(input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024)
            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=1e-6)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=1e-4)
            self.num_epochs_dim2 = num_epochs_dim2
            self.is_target_weights = is_target_weights

            self.device = torch.device("cpu")
            self.stop_during_epochs=True


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
            device=self.device,
            stop_during_epochs=self.stop_during_epochs,
            epoch_thr_for_stopping=11,
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
            device=self.device,
            stop_during_epochs=self.stop_during_epochs,
            epoch_thr_for_stopping=2,
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