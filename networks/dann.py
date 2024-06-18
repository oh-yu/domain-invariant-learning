import torch
from absl import flags
from torch import nn, optim

from ..algo import coral_algo, dann_algo
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
ALGORYTHMS = {
    "DANN": dann_algo,
    "CoRAL": coral_algo,
}


class Dann:
    def __init__(self):
        self.device = torch.device("cpu")

        self.feature_extractor = Conv2d().to(self.device)
        self.task_classifier = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048).to(
            self.device
        )
        self.domain_classifier = ThreeLayersDecoder(input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024).to(
            self.device
        )
        self.domain_criterion = nn.BCELoss()

        self.feature_otimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
        self.domain_optimzier = optim.Adam(self.domain_classifier.parameters(), lr=1e-4)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=1e-4)
        self.num_ecochs = 100

        self.is_target_weights = False
        self.batch_size
    

    def fit_RV(
        self,
        source_ds: torch.utils.data.TensorDataset,
        target_ds: torch.utils.data.TensorDataset,
        test_target_X: torch.Tensor,
        test_target_y_task: torch.Tensor, 
    ) -> float:
    """
    Algorythm: 5.1.2 from https://arxiv.org/abs/1505.07818
    Theory: 3.1 ~ 4.2 from https://link.springer.com/chapter/10.1007/978-3-642-15939-8_35
    """
    # 1. split source into train, val
    train_source_loader, val_source_loader = utils.tensordataset_to_splitted_loaders(source_ds, 16)
    train_target_loader, val_target_loader = utils.tensordataset_to_splitted_loaders(target_ds, 64)
    



    def fit(self, source_loader, target_loader, test_target_X, test_target_y_task):
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
                "criterion": self.domain_criterion,
                "feature_optimizer": self.feature_otimizer,
                "domain_optimizer": self.domain_optimzier,
                "task_optimizer": self.task_optimizer,
            }
            config = {
                "num_epochs": self.num_ecochs,
                "device": self.device,
                "is_changing_lr": True,
                "epoch_thr_for_changing_lr": 11,
                "changed_lrs": [1e-4, 1e-6],
                "stop_during_epochs": True,
                "epoch_thr_for_stopping": 12,
                "is_target_weights": self.is_target_weights,
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier,
                "criterion": self.domain_criterion,
                "feature_optimizer": self.feature_otimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {
                "num_epochs": self.num_ecochs,
                "device": self.device,
                "is_changing_lr": True,
                "epoch_thr_for_changing_lr": 11,
                "changed_lrs": [1e-4],
                "stop_during_epochs": True,
                "epoch_thr_for_stopping": 12,
            }
        self.feature_extractor, self.task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    def predict(self, x):
        return self.task_classifier.predict(self.feature_extractor(x))

    def predict_proba(self, x):
        return self.task_classifier.predict_proba(self.feature_extractor(x))

    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()
