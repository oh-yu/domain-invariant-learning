from absl import flags
import torch
from torch import nn, optim

from ..algo import dann_algo, coral_algo
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_one_layer import OneLayerDecoder
from .mlp_decoder_three_layers import ThreeLayersDecoder

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
            self.is_changing_lr = True

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
            self.is_changing_lr = True

            self.criterion = nn.BCELoss()
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.num_epochs = 300
            self.is_target_weights = True

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
                "is_changing_lr": self.is_changing_lr
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {
                "num_epochs": self.num_epochs,
            }


        self.feature_extractor, self.task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(
            data,
            network,
            **config
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict(self.feature_extractor(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict_proba(self.feature_extractor(x))

    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()
