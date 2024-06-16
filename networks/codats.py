import torch
from absl import flags
from torch import nn, optim
from torch.utils.data import Subset, DataLoader, TensorDataset

from ..algo import coral_algo, dann_algo
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder
from ..utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
ALGORYTHMS = {
    "DANN": dann_algo,
    "CoRAL": coral_algo,
}


class Codats:
    """
    CoDATS model https://arxiv.org/abs/2005.10996
    """

    def __init__(self, experiment: str) -> None:
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.feature_extractor = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.domain_classifier = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.task_classifier = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)

            self.criterion = nn.BCELoss()
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.num_epochs = 300
            self.is_target_weights = True

        elif experiment == "HHAR":
            self.feature_extractor = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.domain_classifier = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(DEVICE)
            self.task_classifier = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)

            self.criterion = nn.BCELoss()
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.num_epochs = 300
            self.is_target_weights = True
    def fit_RV(
        self,
        source_ds: torch.utils.data.TensorDataset,
        target_loader: torch.utils.data.dataloader.DataLoader,
        test_target_X: torch.Tensor,
        test_target_y_task: torch.Tensor, 
    ) -> None:
        # 1. split source into train, val
        N_source = len(source_ds)
        train_idx = [i for i in range(0, N_source//2, 1)]
        val_idx = [i for i in range(N_source//2, N_source, 1)]
        train_source_ds = Subset(source_ds, train_idx)
        val_source_ds = Subset(source_ds, val_idx)

        train_source_loader = DataLoader(train_source_ds, batch_size=34, shuffle=True)
        val_source_loader = DataLoader(val_source_ds, batch_size=34, shuffle=True)

        # 2. free params
        # TODO: Implement

        # 3. RV algo
        ## 3.1 fit f_i
        val_source_X = torch.cat([X for X, _ in val_source_loader], dim=0)
        val_source_y_task = torch.cat([y[:, utils.COL_IDX_TASK] for _, y in val_source_loader], dim=0)
        self.fit(train_source_loader, target_loader, val_source_X, val_source_y_task)

        ## 3.2 fit \bar{f}_i
        target_X = torch.cat([X for X, _ in target_loader], dim=0)
        pred_y_task = self.predict(target_X)
        target_ds = TensorDataset(target_X, pred_y_task)
        target_loader = DataLoader(target_ds, batch_size=34, shuffle=True)
        train_source_X = torch.cat([X for X, _ in train_source_loader], dim=0)
        train_source_ds = TensorDataset(train_source_X)
        train_source_loader = DataLoader(train_source_ds, batch_size=34, shuffle=True)
        self.fit(target_loader, train_source_loader, target_X, pred_y_task)

        ## 3.3 get RV loss
        # TODO: Implement


        # 4. return best
        # TODO: Implement


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
        if FLAGS.algo_name == "DANN":
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
                "is_target_weights": self.is_target_weights,
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {"num_epochs": self.num_epochs}

        self.feature_extractor, self.task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict(self.feature_extractor(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict_proba(self.feature_extractor(x))

    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()
