from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping, get_terminal_weights
from .jdot_algo import _change_lr_during_jdot_training


def fit(data, network, **kwargs):
    """
    Fit Feature Extractor, Task Classifier by DeepJDOT 2D algo.
    TODO: Attach Paper
    """
    # Args
    source_loader, target_loader, target_prime_loader = (
        data["source_loader"],
        data["target_loader"],
        data["target_prime_loader"],
    )
    target_prime_X, target_prime_y_task = data["target_prime_X"], data["target_prime_y_task"]
    feature_extractor, task_classifier = (
        network["feature_extractor"],
        network["task_classifier"],
    )
    criterion = network["criterion"]
    feature_optimizer, task_optimizer = (
        network["feature_optimizer"],
        network["task_optimizer"],
    )
    config = {
        "num_epochs": 1000,
        "device": utils.DEVICE,
        "do_early_stop": False,
        "do_plot": False,
    }
    config.update(kwargs)
    num_epochs, device = config["num_epochs"], config["device"]
    do_early_stop = config["do_early_stop"]
    do_plot = config["do_plot"]
    # Fit
    early_stopping = EarlyStopping()
    loss_domains = []
    loss_tasks = []
    loss_task_evals = []
    loss_pseudo_tasks = []
    num_epochs = torch.tensor(num_epochs, dtype=torch.int32).to(device)
    for epoch in tqdm(range(1, num_epochs.item() + 1)):
        epoch = torch.tensor(epoch, dtype=torch.float32).to(device)
        scheduler = 2 / (1 + torch.exp(-10 * (epoch / (num_epochs + 1)))) - 1
        feature_extractor.train()
        task_classifier.train()
        # batch_training
        for (source_X_batch, source_Y_batch), (target_X_batch, _), (target_prime_X_batch, _) in zip(
            source_loader, target_loader, target_prime_loader
        ):
            # 0. Data
            if task_classifier.output_size == 1:
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK] > 0.5
                source_y_task_batch = source_y_task_batch.to(torch.float32)
            else:
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]
                source_y_task_batch = source_y_task_batch.to(torch.long)

            # 1. Forward
            # 1.1 Feature Extractor
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)
            target_prime_X_batch = feature_extractor(target_prime_X_batch)
            # 1.2 Task Classifier
            pred_source_y_task = task_classifier.predict_proba(source_X_batch)
            pred_target_y_task = task_classifier.predict_proba(target_X_batch)
            pred_target_prime_y_task = task_classifier.predict_proba(target_prime_X_batch)

            # 1.3 Optimal Transport
            ## 1.3.1 Dim1
            loss_domain_mat_dim1 = torch.cdist(target_X_batch, source_X_batch, p=2).to("cpu")
            criterion_pseudo = nn.CrossEntropyLoss(reduction="none")
            if task_classifier.output_size == 1:
                pred_target_y_task = torch.cat(
                    [(1 - pred_target_y_task).reshape(-1, 1), pred_target_y_task.reshape(-1, 1)], dim=1
                )
            num_classes = pred_target_y_task.shape[1]
            pred_target_y_task = pred_target_y_task.unsqueeze(0).expand(len(source_y_task_batch), -1, -1)
            source_y_task_batch_expanded = source_y_task_batch.unsqueeze(1).expand(-1, pred_target_y_task.shape[1])
            loss_pseudo_task_mat_dim1 = (
                criterion_pseudo(
                    pred_target_y_task.reshape(-1, num_classes), source_y_task_batch_expanded.reshape(-1).to(torch.long)
                )
                .reshape(len(source_y_task_batch_expanded), pred_target_y_task.shape[1])
                .T
            )
            loss_pseudo_task_mat_dim1 = loss_pseudo_task_mat_dim1.to("cpu")
            cost_mat_dim1 = loss_domain_mat_dim1 + loss_pseudo_task_mat_dim1
            optimal_transport_weights_dim1 = ot.emd(
                torch.ones(len(target_X_batch)) / len(target_X_batch),
                torch.ones(len(source_X_batch)) / len(source_X_batch),
                cost_mat_dim1,
            )

            ## 1.3.2 Dim2
            loss_domain_mat_dim2 = torch.cdist(target_prime_X_batch, target_X_batch, p=2).to("cpu")
            if task_classifier.output_size == 1:
                pred_target_prime_y_task = torch.cat(
                    [(1 - pred_target_prime_y_task).reshape(-1, 1), pred_target_prime_y_task.reshape(-1, 1)], dim=1
                )
            pred_target_prime_y_task = pred_target_prime_y_task.unsqueeze(0).expand(len(source_y_task_batch), -1, -1)
            source_y_task_batch_expanded = source_y_task_batch.unsqueeze(1).expand(
                -1, pred_target_prime_y_task.shape[1]
            )
            loss_pseudo_task_mat_dim2 = (
                criterion_pseudo(
                    pred_target_prime_y_task.reshape(-1, num_classes),
                    source_y_task_batch_expanded.reshape(-1).to(torch.long),
                )
                .reshape(len(source_y_task_batch_expanded), pred_target_prime_y_task.shape[1])
                .T
            )
            if target_X_batch.shape[0] <= source_X_batch.shape[0]:
                loss_pseudo_task_mat_dim2 = loss_pseudo_task_mat_dim2.to("cpu")[:, : loss_domain_mat_dim2.shape[1]]
                cost_mat_dim2 = loss_domain_mat_dim2 + loss_pseudo_task_mat_dim2
                optimal_transport_weights_dim2 = ot.emd(
                    torch.ones(len(target_prime_X_batch)) / len(target_prime_X_batch),
                    torch.ones(len(target_X_batch)) / len(target_X_batch),
                    cost_mat_dim2,
                )

            else:
                loss_domain_mat_dim2 = loss_domain_mat_dim2[:, : source_X_batch.shape[0]]
                loss_pseudo_task_mat_dim2 = loss_pseudo_task_mat_dim2.to("cpu")
                cost_mat_dim2 = loss_domain_mat_dim2 + loss_pseudo_task_mat_dim2
                optimal_transport_weights_dim2 = ot.emd(
                    torch.ones(len(target_prime_X_batch)) / len(target_prime_X_batch),
                    torch.ones(len(source_X_batch)) / len(source_X_batch),
                    cost_mat_dim2,
                )

            # 1.4 Align Loss
            loss_domain = torch.mean(optimal_transport_weights_dim1 * loss_domain_mat_dim1)
            loss_domain += torch.mean(optimal_transport_weights_dim2 * loss_domain_mat_dim2)
            loss_domains.append(loss_domain.item())

            # 1.5 Task Loss
            loss_pseudo_task = torch.mean(optimal_transport_weights_dim1 * loss_pseudo_task_mat_dim1)
            loss_pseudo_task += torch.mean(optimal_transport_weights_dim2 * loss_pseudo_task_mat_dim2)
            loss_pseudo_tasks.append(loss_pseudo_task.item())

            if task_classifier.output_size == 1:
                criterion_weight = nn.BCELoss()
                loss_task = criterion_weight(pred_source_y_task, source_y_task_batch)
                loss_tasks.append(loss_task.item())
            else:
                criterion_weight = nn.CrossEntropyLoss(reduction="none")
                loss_task = criterion_weight(pred_source_y_task, source_y_task_batch)
                loss_task = loss_task
                loss_task = loss_task.mean()
                loss_tasks.append(loss_task.item())

            # 2. Backward
            loss = loss_task + scheduler * loss_domain + loss_pseudo_task
            feature_optimizer.zero_grad()
            task_classifier.zero_grad()
            loss.backward()

            # 3. Update Params
            feature_optimizer.step()
            task_optimizer.step()

        # 4. Eval
        feature_extractor.eval()
        task_classifier.eval()
        with torch.no_grad():
            target_prime_feature_eval = feature_extractor(target_prime_X)
            pred_y_task_eval = task_classifier.predict(target_prime_feature_eval)
            acc = sum(pred_y_task_eval == target_prime_y_task) / target_prime_y_task.shape[0]
            early_stopping(acc)
        loss_task_evals.append(acc.item())
        if early_stopping.early_stop & do_early_stop:
            break
        print(
            f"Epoch: {epoch}, Loss Domain: {loss_domain}, Loss Task: {loss_task}, Loss Pseudo Task: {loss_pseudo_task}, Acc: {acc}"
        )
    if do_plot:
        plt.figure()
        plt.plot(loss_domains, label="loss_domain")
        plt.plot(loss_pseudo_tasks, label="loss_pseudo_task")
        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.legend()

        plt.figure()
        plt.plot(loss_tasks, label="loss_task")
        plt.xlabel("batch")
        plt.ylabel("cross entropy loss")
        plt.legend()

        plt.figure()
        plt.plot(loss_task_evals, label="loss_task_eval")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()

    return feature_extractor, task_classifier, loss_task_evals
