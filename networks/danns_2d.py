import numpy as np
import torch
from absl import flags
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..algo import coral2D_algo, dann2D_algo, jdot2D_algo
from ..utils import utils
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
ALGORYTHMS = {"DANN": dann2D_algo, "CoRAL": coral2D_algo, "JDOT": jdot2D_algo}


class Danns2D:
    def __init__(self, experiment: str):
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR", "MNIST"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.feature_extractor = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.domain_classifier_dim2 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.task_classifier = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)

            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs = 300
            self.device = DEVICE
            self.batch_size = 32
            self.experiment = experiment
            self.do_early_stop = False

        elif experiment == "HHAR":
            self.feature_extractor = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                DEVICE
            )
            self.domain_classifier_dim2 = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                DEVICE
            )
            self.task_classifier = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs = 300
            self.device = DEVICE
            self.batch_size = 128
            self.experiment = experiment
            self.do_early_stop = False

        elif experiment in ["MNIST"]:
            self.feature_extractor = Conv2d()
            self.task_classifier = ThreeLayersDecoder(input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048)
            self.domain_classifier_dim1 = ThreeLayersDecoder(
                input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024
            )
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs = 100

            self.domain_classifier_dim2 = ThreeLayersDecoder(
                input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024
            )
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.0001)
            self.device = torch.device("cpu")
            self.batch_size = 16
            self.experiment = experiment
            self.do_early_stop = False

    def fit(self, source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task):
        if FLAGS.is_RV_tuning:
            return self._fit_RV(
                source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task
            )
        else:
            return self._fit(
                source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task
            )

    def _fit_RV(self, source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task):
        """
        Algorythm, Proof: https://drive.google.com/file/d/1YkNMMKeOY4P-HfL2G5GgIrnRJD-lYY96/view?usp=sharing
        Theory: 3.1 ~ 4.2 from https://link.springer.com/chapter/10.1007/978-3-642-15939-8_35
        """
        # S -> S', S_V
        source_X = torch.cat([X for X, _ in source_loader], dim=0)
        source_y_task = torch.cat([y for _, y in source_loader], dim=0)
        source_ds = TensorDataset(source_X, source_y_task)
        train_source_loader, val_source_loader = utils.tensordataset_to_splitted_loaders(source_ds, self.batch_size)

        free_params = [
            {"lr": 0.00001, "eps": 1e-08, "weight_decay": 0},
            {"lr": 0.0001, "eps": 1e-08, "weight_decay": 0},
            {"lr": 0.001, "eps": 1e-08, "weight_decay": 0},
        ]
        RV_scores = {"free_params": [], "scores": []}

        for param in free_params:
            # Fit eta
            self.__init__(self.experiment)
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer_dim1.param_groups[0].update(param)
            self.domain_optimizer_dim2.param_groups[0].update(param)
            self.task_optimizer.param_groups[0].update(param)

            val_source_X = torch.cat([X for X, _ in val_source_loader], dim=0)
            val_source_y_task = torch.cat([y[:, utils.COL_IDX_TASK] for _, y in val_source_loader], dim=0)
            self.do_early_stop = True
            self._fit(train_source_loader, target_loader, target_prime_loader, val_source_X, val_source_y_task)
            # Fit eta_r
            target_prime_X = torch.cat([X for X, _ in target_prime_loader], dim=0)
            pred_y_task = self.predict(target_prime_X)
            target_prime_ds = TensorDataset(
                target_prime_X,
                torch.cat(
                    [pred_y_task.reshape(-1, 1), torch.zeros_like(pred_y_task).reshape(-1, 1).to(torch.float32)], dim=1
                ),
            )
            target_prime_as_source_loader = DataLoader(target_prime_ds, batch_size=self.batch_size, shuffle=True)

            train_source_X = torch.cat([X for X, _ in train_source_loader], dim=0)
            train_source_ds = TensorDataset(
                train_source_X, torch.ones(train_source_X.shape[0]).to(torch.float32).to(self.device)
            )
            train_source_as_target_prime_loader = DataLoader(train_source_ds, batch_size=self.batch_size, shuffle=True)
            self.__init__(self.experiment)
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer_dim1.param_groups[0].update(param)
            self.domain_optimizer_dim2.param_groups[0].update(param)
            self.task_optimizer.param_groups[0].update(param)
            self.do_early_stop = True
            self._fit(
                target_prime_as_source_loader,
                target_loader,
                train_source_as_target_prime_loader,
                target_prime_X,
                pred_y_task,
            )

            # Get RV Loss
            pred_y_task = self.predict(val_source_X)
            acc_RV = sum(pred_y_task == val_source_y_task) / len(pred_y_task)
            RV_scores["free_params"].append(param)
            RV_scores["scores"].append(acc_RV.item())

        # Retraining
        best_param = RV_scores["free_params"][np.argmax(RV_scores["scores"])]
        self.__init__(self.experiment)
        self.feature_optimizer.param_groups[0].update(best_param)
        self.domain_optimizer_dim1.param_groups[0].update(best_param)
        self.domain_optimizer_dim2.param_groups[0].update(best_param)
        self.task_optimizer.param_groups[0].update(best_param)
        if self.experiment == "MNIST":
            self.do_early_stop = True
        else:
            self.do_early_stop = False
        self._fit(source_loader, target_loader, target_prime_loader, val_source_X, val_source_y_task)
        pred_y_task = self.predict(test_target_prime_X)
        acc = sum(pred_y_task == test_target_prime_y_task) / pred_y_task.shape[0]
        return acc.item()

    def _fit(self, source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task):
        data = {
            "source_loader": source_loader,
            "target_loader": target_loader,
            "target_prime_loader": target_prime_loader,
            "target_prime_X": test_target_prime_X,
            "target_prime_y_task": test_target_prime_y_task,
        }
        network = {
            "feature_extractor": self.feature_extractor,
            "domain_classifier_dim1": self.domain_classifier_dim1,
            "domain_classifier_dim2": self.domain_classifier_dim2,
            "task_classifier": self.task_classifier,
            "criterion": self.criterion,
            "feature_optimizer": self.feature_optimizer,
            "domain_optimizer_dim1": self.domain_optimizer_dim1,
            "domain_optimizer_dim2": self.domain_optimizer_dim2,
            "task_optimizer": self.task_optimizer,
        }
        config = {"num_epochs": self.num_epochs, "device": self.device, "do_early_stop": self.do_early_stop}
        self.feature_extractor, self.task_classifier, acc = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)
        return acc

    def predict(self, X):
        out = self.feature_extractor(X)
        out = self.task_classifier.predict(out)
        return out
