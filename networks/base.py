from abc import ABC, abstractmethod
from absl import flags
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ..algo import coral_algo, dann_algo, jdot_algo, supervised_algo
from ..utils import utils


FLAGS = flags.FLAGS
ALGORYTHMS = {"DANN": dann_algo, "CoRAL": coral_algo, "JDOT": jdot_algo}


class DannsBase(ABC):
    def __init__(self, experiment: str) -> None:
        pass

    def fit(
        self,
        source_ds: torch.utils.data.TensorDataset,
        target_ds: torch.utils.data.TensorDataset,
        test_target_X: torch.Tensor,
        test_target_y_task: torch.Tensor,
    ):
        if FLAGS.is_RV_tuning:
            return self._fit_RV(source_ds, target_ds, test_target_X, test_target_y_task)
        else:
            source_loader = DataLoader(source_ds, batch_size=self.batch_size, shuffle=True)
            target_loader = DataLoader(target_ds, batch_size=self.batch_size, shuffle=True)
            self._fit(source_loader, target_loader, test_target_X, test_target_y_task)
            self.set_eval()
            pred_y_task = self.predict(test_target_X)
            acc = sum(pred_y_task == test_target_y_task) / len(pred_y_task)
            return acc.item()

    def _fit_RV(
        self,
        source_ds: torch.utils.data.TensorDataset,
        target_ds: torch.utils.data.TensorDataset,
        test_target_X: torch.Tensor,
        test_target_y_task: torch.Tensor,
    ) -> float:
        """
        Algorythm: 5.1.2 from https://arxiv.org/abs/1505.07818
        Theory: 3.1 ~ 4.2 from https://link.springer.com/chapter/10.1007/978-3-642-15939-8_35
        Proof: https://drive.google.com/file/d/1BLldo_Kun1Kx_Hhq1o2IJF7ssKXGKoCX/view?usp=sharing
        """
        # 1. split source into train, val
        train_source_loader, val_source_loader = utils.tensordataset_to_splitted_loaders(source_ds, self.batch_size)
        train_target_loader, val_target_loader = utils.tensordataset_to_splitted_loaders(target_ds, self.batch_size)

        # 2. free params
        free_params = [
            {"lr": 0.00001, "eps": 1e-08, "weight_decay": 0},
            {"lr": 0.0001, "eps": 1e-08, "weight_decay": 0},
            {"lr": 0.001, "eps": 1e-08, "weight_decay": 0},
        ]
        RV_scores = {"free_params": [], "scores": []}
        for param in free_params:
            self.__init__(self.experiment)
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer.param_groups[0].update(param)
            self.task_optimizer.param_groups[0].update(param)

            # 3. RV algo
            ## 3.1 fit f_i
            val_source_X = torch.cat([X for X, _ in val_source_loader], dim=0)
            val_source_y_task = torch.cat([y[:, utils.COL_IDX_TASK] for _, y in val_source_loader], dim=0)
            self.do_early_stop = True
            self._fit(train_source_loader, train_target_loader, val_source_X, val_source_y_task)

            ## 3.2 fit \bar{f}_i
            train_target_X = torch.cat([X for X, _ in train_target_loader], dim=0)
            train_target_pred_y_task = self.predict(train_target_X)
            val_target_X = torch.cat([X for X, _ in val_target_loader], dim=0)
            val_target_pred_y_task = self.predict(val_target_X)

            train_target_ds = TensorDataset(
                train_target_X,
                torch.cat(
                    [
                        train_target_pred_y_task.reshape(-1, 1),
                        torch.zeros_like(train_target_pred_y_task).reshape(-1, 1).to(torch.float32),
                    ],
                    dim=1,
                ),
            )
            target_as_source_loader = DataLoader(train_target_ds, batch_size=self.batch_size, shuffle=True)

            train_source_X = torch.cat([X for X, _ in train_source_loader], dim=0)
            train_source_ds = TensorDataset(
                train_source_X, torch.ones(train_source_X.shape[0]).to(torch.float32).to(self.device)
            )
            train_source_as_target_loader = DataLoader(train_source_ds, batch_size=self.batch_size, shuffle=True)
            self.__init__(self.experiment)
            self.feature_optimizer.param_groups[0].update(param)
            self.domain_optimizer.param_groups[0].update(param)
            self.task_optimizer.param_groups[0].update(param)
            self.do_early_stop = True
            self._fit(target_as_source_loader, train_source_as_target_loader, val_target_X, val_target_pred_y_task)

            ## 3.3 get RV loss
            pred_y_task = self.predict(val_source_X)
            acc_RV = sum(pred_y_task == val_source_y_task) / val_source_y_task.shape[0]

            # 3.4 get terminal evaluation
            RV_scores["free_params"].append(param)
            RV_scores["scores"].append(acc_RV.item())

        # 4. Retraining
        best_param = RV_scores["free_params"][np.argmax(RV_scores["scores"])]
        self.__init__(self.experiment)
        self.feature_optimizer.param_groups[0].update(best_param)
        self.domain_optimizer.param_groups[0].update(best_param)
        self.task_optimizer.param_groups[0].update(best_param)
        source_loader = DataLoader(source_ds, batch_size=self.batch_size, shuffle=True)
        target_loader = DataLoader(target_ds, batch_size=self.batch_size, shuffle=True)
        self.do_early_stop = False
        self._fit(source_loader, target_loader, val_source_X, val_source_y_task)
        self.set_eval()
        pred_y_task = self.predict(test_target_X)
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        return acc.item()

    def _fit(
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
                "do_early_stop": self.do_early_stop,
                "device": self.device,
            }
        elif FLAGS.algo_name == "CoRAL":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {"num_epochs": self.num_epochs, "do_early_stop": self.do_early_stop, "device": self.device}
        elif FLAGS.algo_name == "JDOT":
            network = {
                "feature_extractor": self.feature_extractor,
                "task_classifier": self.task_classifier,
                "criterion": self.criterion,
                "feature_optimizer": self.feature_optimizer,
                "task_optimizer": self.task_optimizer,
            }
            config = {"num_epochs": self.num_epochs, "do_early_stop": self.do_early_stop, "device": self.device}
        self.feature_extractor, self.task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict(self.feature_extractor(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_classifier.predict_proba(self.feature_extractor(x))

    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()


class SupervisedBase(nn.Module):
    def __init__(self):
        super().__init__()

    def fit_without_adapt(self, source_loader):
        data = {"loader": source_loader}
        network = {
            "decoder": self.decoder,
            "encoder": self.encoder,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
        }
        config = {"use_source_loader": True, "num_epochs": self.num_epochs}
        supervised_algo.fit(data, network, **config)

    def fit_on_target(self, train_target_prime_loader):
        data = {"loader": train_target_prime_loader}
        network = {
            "decoder": self.decoder,
            "encoder": self.encoder,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
        }
        config = {
            "use_source_loader": False,
            "num_epochs": self.num_epochs,
        }
        supervised_algo.fit(data, network, **config)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def predict(self, x):
        return self.decoder.predict(self.encoder(x))

    def predict_proba(self, x):
        return self.decoder.predict_proba(self.encoder(x))
