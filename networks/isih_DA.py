from absl import flags
import torch
from torch import nn, optim

from ..algo import dann_algo, coral_algo
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .conv2d import Conv2d
from .mlp_decoder_one_layer import OneLayerDecoder
from .mlp_decoder_three_layers import ThreeLayersDecoder

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
                "epoch_thr_for_stopping": 11
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
                "device": self.device
            }
        self.feature_extractor, self.task_classifier_dim1, _ = ALGORYTHMS[FLAGS.algo_name].fit(
            data,
            network,
            **config
        )

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
                "epoch_thr_for_stopping": 2
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
                "device": self.device
            }      

        self.feature_extractor, self.task_classifier_dim2, _ = ALGORYTHMS[FLAGS.algo_name].fit(
            data,
            network,
            **config
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
