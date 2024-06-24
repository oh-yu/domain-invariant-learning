import numpy as np
import torch
from absl import flags
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from ..algo import coral_algo, dann_algo
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder
from ..utils import utils

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

        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
        self.domain_optimizer = optim.Adam(self.domain_classifier.parameters(), lr=1e-4)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=1e-4)
        self.num_ecochs = 100

        self.is_target_weights = False


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
        train_source_loader, val_source_loader = utils.tensordataset_to_splitted_loaders(source_ds, 64)
        train_target_loader, val_target_loader = utils.tensordataset_to_splitted_loaders(target_ds, 64)

        # 2. free params
        free_params = [
            {"learning_rate": 0.0001, "eps": 1e-08, "weight_decay": 0},
            {"learning_rate": 0.001, "eps": 1e-08, "weight_decay": 0},
            {"learning_rate": 0.01, "eps": 1e-08, "weight_decay": 0},
        ]
        RV_scores = {"free_params": [], "scores": []}
        for param in free_params:
            self.__init__()
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer.param_groups[0].update(param)
            self.task_optimizer.param_groups[0].update(param)

            # 3. RV algo
            ## 3.1 fit f_i
            val_source_X = torch.cat([X for X, _ in val_source_loader], dim=0)
            val_source_y_task = torch.cat([y[:, utils.COL_IDX_TASK] for _, y in val_source_loader], dim=0)
            self.do_early_stop = True
            self.fit(train_source_loader, train_target_loader, val_source_X, val_source_y_task)

            ## 3.2 fit \bar{f}_i
            train_target_X = torch.cat([X for X, _ in train_target_loader], dim=0)
            train_target_pred_y_task = self.predict(train_target_X)
            val_target_X = torch.cat([X for X, _ in val_target_loader], dim=0)
            val_target_pred_y_task = self.predict(val_target_X)

            train_target_ds = TensorDataset(train_target_X, torch.cat([train_target_pred_y_task.reshape(-1, 1), torch.zeros_like(train_target_pred_y_task).reshape(-1, 1).to(torch.float32)], dim=1))
            target_as_source_loader = DataLoader(train_target_ds, batch_size=64, shuffle=True)

            train_source_X = torch.cat([X for X, _ in train_source_loader], dim=0)
            train_source_ds = TensorDataset(train_source_X, torch.ones(train_source_X.shape[0]).to(torch.float32))
            train_source_as_target_loader = DataLoader(train_source_ds, batch_size=64, shuffle=True)
            self.__init__()
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer.param_groups[0].update(param)
            self.task_optimizer.param_groups[0].update(param)
            self.do_early_stop = True
            self.fit(target_as_source_loader, train_source_as_target_loader, val_target_X, val_target_pred_y_task)

            ## 3.3 get RV loss
            pred_y_task = self.predict(val_source_X)
            acc_RV = sum(pred_y_task == val_source_y_task) / val_source_y_task.shape[0]


            # 3.4 get terminal evaluation
            RV_scores["free_params"].append(param)
            RV_scores["scores"].append(acc_RV.item())

        # 4. Retraining
        best_param = RV_scores["free_params"][np.argmax(RV_scores["scores"])]
        self.__init__()
        self.feature_optimizer.param_groups[0].update(best_param)
        self.domain_optimizer.param_groups[0].update(best_param)
        self.task_optimizer.param_groups[0].update(best_param)
        source_loader = DataLoader(source_ds, batch_size=64, shuffle=True)
        target_loader = DataLoader(target_ds, batch_size=64, shuffle=True)
        self.do_early_stop = False
        self.fit(source_loader, target_loader, val_source_X, val_source_y_task)
        self.set_eval()
        pred_y_task = self.predict(test_target_X)
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        return acc.item()


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
                "feature_optimizer": self.feature_optimizer,
                "domain_optimizer": self.domain_optimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {
                "num_epochs": self.num_ecochs,
                "device": self.device,
                "is_target_weights": self.is_target_weights,
                "do_early_stop": True
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier,
                "criterion": self.domain_criterion,
                "feature_optimizer": self.feature_optimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {
                "num_epochs": self.num_ecochs,
                "device": self.device,
            }
        self.feature_extractor, self.task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    def predict(self, x):
        return self.task_classifier.predict(self.feature_extractor(x))

    def predict_proba(self, x):
        return self.task_classifier.predict_proba(self.feature_extractor(x))

    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()
