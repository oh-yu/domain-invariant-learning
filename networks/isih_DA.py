import torch
from absl import flags
from torch import nn, optim
from torch.utils.data import Subset, DataLoader, TensorDataset

from ..algo import coral_algo, dann_algo
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder
from ..utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
ALGORYTHMS = {
    "DANN": dann_algo,
    "CoRAL": coral_algo,
}


class IsihDanns:
    """
    TODO: Attach paper
    """

    def __init__(self, experiment: str):
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR", "MNIST"]

        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.feature_extractor = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.task_classifier_dim1 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)

            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = 200

            self.domain_classifier_dim2 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.task_classifier_dim2 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=0.00005)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.00005)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=0.00005)
            self.num_epochs_dim2 = 100
            self.is_target_weights = True

            self.device = DEVICE
            self.stop_during_epochs = False

            self.batch_size = 34

        elif experiment == "HHAR":
            self.feature_extractor = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                DEVICE
            )
            self.task_classifier_dim1 = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)
            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = 200

            self.domain_classifier_dim2 = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                DEVICE
            )
            self.task_classifier_dim2 = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)

            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=0.00005)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.00005)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=0.00005)
            self.num_epochs_dim2 = 100
            self.is_target_weights = True

            self.device = DEVICE
            self.stop_during_epochs = False
            self.batch_size = 128

        elif experiment in ["MNIST"]:
            self.feature_extractor = Conv2d()
            self.task_classifier_dim1 = ThreeLayersDecoder(
                input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048
            )
            self.domain_classifier_dim1 = ThreeLayersDecoder(
                input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024
            )
            self.feature_optimizer_dim1 = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.task_optimizer_dim1 = optim.Adam(self.task_classifier_dim1.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs_dim1 = 100

            self.task_classifier_dim2 = ThreeLayersDecoder(
                input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048
            )
            self.domain_classifier_dim2 = ThreeLayersDecoder(
                input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024
            )
            self.feature_optimizer_dim2 = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=1e-6)
            self.task_optimizer_dim2 = optim.Adam(self.task_classifier_dim2.parameters(), lr=1e-4)
            self.num_epochs_dim2 = 100
            self.is_target_weights = False

            self.device = torch.device("cpu")
            self.stop_during_epochs = True
            self.batch_size = 64

    def fit_RV_1st_dim(self, source_ds: torch.utils.data.TensorDataset,  target_loader: torch.utils.data.dataloader.DataLoader, test_target_X: torch.Tensor, test_target_y_task: torch.Tensor) -> None:
        # 1. split source into train, val
        N_source = len(source_ds)
        train_idx = [i for i in range(0, N_source//2, 1)]
        val_idx = [i for i in range(N_source//2, N_source, 1)]
        train_source_ds = Subset(source_ds, train_idx)
        val_source_ds = Subset(source_ds, val_idx)

        train_source_loader = DataLoader(train_source_ds, batch_size=self.batch_size, shuffle=True)
        val_source_loader = DataLoader(val_source_ds, batch_size=self.batch_size, shuffle=True)
        # 2. free params
        free_params = [
            {"learning_rate": 0.0001, "eps": 1e-08, "weight_decay": 0},
            {"learning_rate": 0.001, "eps": 1e-08, "weight_decay": 0},
            {"learning_rate": 0.01, "eps": 1e-08, "weight_decay": 0},
            {"learning_rate": 0.0001, "eps": 1e-04, "weight_decay": 0},
            {"learning_rate": 0.0001, "eps": 1e-08, "weight_decay": 0.1},
        ]
        RV_scores = {"free_params": [], "scores": [], "terminal_acc": []}
        for param in free_params:
            self.__init__(self.experiment)
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer_dim1.param_groups[0].update(param)
            self.task_optimizer_dim1.param_groups[0].update(param)
            # 3. RV algo
            ## 3.1 fit f_i
            val_source_X = torch.cat([X for X, _ in val_source_loader], dim=0)
            val_source_y_task = torch.cat([y[:, utils.COL_IDX_TASK] for _, y in val_source_loader], dim=0)
            self.fit_1st_dim(train_source_loader, target_loader, val_source_X, val_source_y_task)
            ## 3.2 fit \bar{f}_i
            target_X = torch.cat([X for X, _ in target_loader], dim=0)
            pred_y_task = self.predict(target_X)
            target_ds = TensorDataset(target_X, torch.cat([pred_y_task.reshape(-1, 1), torch.zeros_like(pred_y_task).reshape(-1, 1).to(torch.float32)], dim=1))
            target_as_source_loader = DataLoader(target_ds, batch_size=self.batch_size, shuffle=True)

            train_source_X = torch.cat([X for X, _ in train_source_loader], dim=0)
            train_source_ds = TensorDataset(train_source_X, torch.ones(train_source_X.shape[0]).to(torch.float32).to(utils.DEVICE))
            train_source_as_target_loader = DataLoader(train_source_ds, batch_size=self.batch_size, shuffle=True)
    
            self.__init__(self.experiment)
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer_dim1.param_groups[0].update(param)
            self.task_optimizer_dim1.param_groups[0].update(param)

            self.fit_1st_dim(target_as_source_loader, train_source_as_target_loader, target_X, pred_y_task)
            ## 3.3 get RV loss
            ## 3.4 get terminal evaluation

        # 4. Retraining
        pass


    def fit_1st_dim(self, source_loader, target_loader, test_target_X: torch.Tensor, test_target_y_task: torch.Tensor):
        data = {
            "source_loader": source_loader,
            "target_loader": target_loader,
            "target_X": test_target_X,
            "target_y_task": test_target_y_task,
        }
        if FLAGS.algo_name == "DANN":
            network = {
                "feature_extractor": self.feature_extractor,
                "domain_classifier": self.domain_classifier_dim1,
                "task_classifier": self.task_classifier_dim1,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer_dim1,
                "domain_optimizer": self.domain_optimizer_dim1,
                "task_optimizer": self.task_optimizer_dim1,
            }
            config = {
                "num_epochs": self.num_epochs_dim1,
                "is_target_weights": self.is_target_weights,
                "device": self.device,
                "stop_during_epochs": self.stop_during_epochs,
                "epoch_thr_for_stopping": 11,
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier_dim1,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer_dim1,
                "task_optimizer": self.task_optimizer_dim1,
            }
            config = {
                "num_epochs": self.num_epochs_dim1,
                "device": self.device,
                "stop_during_epochs": self.stop_during_epochs,
                "epoch_thr_for_stopping": 11,
            }
        self.feature_extractor, self.task_classifier_dim1, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    def fit_2nd_dim(self, source_loader, target_loader, test_target_X: torch.Tensor, test_target_y_task: torch.Tensor):
        data = {
            "source_loader": source_loader,
            "target_loader": target_loader,
            "target_X": test_target_X,
            "target_y_task": test_target_y_task,
        }
        if FLAGS.algo_name == "DANN":
            network = {
                "feature_extractor": self.feature_extractor,
                "domain_classifier": self.domain_classifier_dim2,
                "task_classifier": self.task_classifier_dim2,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer_dim2,
                "domain_optimizer": self.domain_optimizer_dim2,
                "task_optimizer": self.task_optimizer_dim2,
            }
            config = {
                "num_epochs": self.num_epochs_dim2,
                "is_psuedo_weights": True,
                "is_target_weights": self.is_target_weights,
                "device": self.device,
                "stop_during_epochs": self.stop_during_epochs,
                "epoch_thr_for_stopping": 2,
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier_dim2,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer_dim2,
                "task_optimizer": self.task_optimizer_dim2,
            }
            config = {
                "num_epochs": self.num_epochs_dim2,
                "is_psuedo_weights": True,
                "device": self.device,
                "stop_during_epochs": self.stop_during_epochs,
                "epoch_thr_for_stopping": 2,
            }

        self.feature_extractor, self.task_classifier_dim2, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

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
