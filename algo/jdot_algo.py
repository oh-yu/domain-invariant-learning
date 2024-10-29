from typing import List

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping, get_psuedo_label_weights, get_terminal_weights


def fit(data, network, **kwargs):
    """
    Fit Feature Extractor, Task Classifier by DeepJDOT algo.
    https://arxiv.org/abs/1803.10081
    """
    # Args
    source_loader, target_loader = data["source_loader"], data["target_loader"]
    target_X, target_y_task = data["target_X"], data["target_y_task"]

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
        "is_target_weights": False,
        "is_class_weights": False,
        "is_psuedo_weights": False,
        "do_plot": False,
        "do_print": False,
        "device": utils.DEVICE,
        "is_changing_lr": False,
        "epoch_thr_for_changing_lr": 200,
        "changed_lrs": [0.00005, 0.00005],
        "stop_during_epochs": False,
        "epoch_thr_for_stopping": 2,
        "do_early_stop": False,
    }
    config.update(kwargs)
    num_epochs = config["num_epochs"]
    is_target_weights, is_class_weights, is_psuedo_weights = (
        config["is_target_weights"],
        config["is_class_weights"],
        config["is_psuedo_weights"],
    )
    do_plot, _ = config["do_plot"], config["do_print"]
    device = config["device"]
    is_changing_lr, epoch_thr_for_changing_lr, changed_lrs = (
        config["is_changing_lr"],
        config["epoch_thr_for_changing_lr"],
        config["changed_lrs"],
    )
    stop_during_epochs, epoch_thr_for_stopping = config["stop_during_epochs"], config["epoch_thr_for_stopping"]
    do_early_stop = config["do_early_stop"]
    # Fit
    early_stopping = EarlyStopping()
    loss_domains = []
    loss_tasks = []
    loss_task_evals = []
    loss_peudo_tasks = []
    num_epochs = torch.tensor(num_epochs, dtype=torch.int32).to(device)
    for epoch in tqdm(range(1, num_epochs.item() + 1)):
        epoch = torch.tensor(epoch, dtype=torch.float32).to(device)
        feature_extractor.train()
        task_classifier.train()
        if stop_during_epochs & (epoch.item() == epoch_thr_for_stopping):
            break
        if is_changing_lr:
            feature_optimizer, task_optimizer = _change_lr_during_jdot_training(
                feature_optimizer,
                task_optimizer,
                epoch,
                epoch_thr=epoch_thr_for_changing_lr,
                changed_lrs=changed_lrs,
            )
        # batch_training
        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(
            source_loader, target_loader
        ):
            # 0. Data
            if task_classifier.output_size == 1:
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK] > 0.5
                source_y_task_batch = source_y_task_batch.to(torch.float32)
            else:
                if is_psuedo_weights:
                    output_size = source_Y_batch[:, :-1].shape[1]
                    source_y_task_batch = source_Y_batch[:, :output_size]
                    source_y_task_batch = torch.argmax(source_y_task_batch, dim=1)
                    source_y_task_batch = source_y_task_batch.to(torch.long)
                else:
                    source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]
                    source_y_task_batch = source_y_task_batch.to(torch.long)
            if is_psuedo_weights:
                weights = get_psuedo_label_weights(source_Y_batch=source_Y_batch, device=device).detach()
            else:
                weights = torch.ones_like(source_y_task_batch)
            # 1. Forward
            # 1.1 Feature Extractor
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)
            # 1.2 Task Classifier
            pred_source_y_task = task_classifier.predict_proba(source_X_batch)
            pred_target_y_task = task_classifier.predict_proba(target_X_batch)

            # 1.3 Optimal Transport
            loss_domain_mat = torch.cdist(target_X_batch, source_X_batch, p=2)
            
            if task_classifier.output_size == 1:
                criterion_pseudo = nn.BCELoss()
                loss_pseudo_task_mat = torch.ones((target_X_batch.shape[0], source_X_batch.shape[0]))
                for i in range(pred_target_y_task.shape[0]):
                    for j in range(source_y_task_batch.shape[0]):
                        loss_pseudo_task = criterion_pseudo(pred_target_y_task[i], source_y_task_batch[j])
                        loss_pseudo_task_mat[i][j] = loss_pseudo_task
            else:
                criterion_pseudo = nn.BCELoss()
                loss_pseudo_task_mat = torch.ones((target_X_batch.shape[0], source_X_batch.shape[0]))
                for i in range(pred_target_y_task.shape[0]):
                    for j in range(source_y_task_batch.shape[0]):
                        loss_pseudo_task = criterion_pseudo(pred_target_y_task[i], source_y_task_batch[j])
                        loss_pseudo_task_mat[i][j] = loss_pseudo_task
            """
            Intuitve Sense

            Min: Σ_{i, j} c_ij * x_ij
            s.t. 
            (1) Σ_i x_ij = a_j, j∈{0, 1, 2, ..., M}
            (2) Σ_j x_ij = b_i, i∈{0, 1, 2, ..., N}

            x_ij: amount of flow from i to j
            c_ij: cost(=distance between vectors) of flow from i to j
            a_j: demand of j(uniform)
            b_i: supply of i(uniform)
            """
            cost_mat = loss_domain_mat + loss_pseudo_task_mat
            optimal_transport_weights = ot.emd2(torch.ones(len(target_X_batch)) / len(target_X_batch), torch.ones(len(source_X_batch)) / len(source_X_batch), cost_mat)

            # 1.4 Align Loss
            loss_domain = torch.mean(optimal_transport_weights*loss_domain_mat)

            # 1.5 Task Loss
            loss_pseudo_task = torch.mean(optimal_transport_weights*loss_pseudo_task_mat)
            if task_classifier.output_size == 1:
                criterion_weight = nn.BCELoss(weight=weights.detach())
                loss_task = criterion_weight(pred_source_y_task, source_y_task_batch)
                loss_tasks.append(loss_task.item())
            else:
                criterion_weight = nn.CrossEntropyLoss(reduction="none")
                loss_task = criterion_weight(pred_source_y_task, source_y_task_batch)
                loss_task = loss_task * weights
                loss_task = loss_task.mean()
                loss_tasks.append(loss_task.item())
            
            # 2. Backward
            loss = loss_task + loss_domain + loss_pseudo_task
            feature_optimizer.zero_grad()
            task_optimizer.zero_grad()
            loss.backward()
            # 3. Update Params
            feature_optimizer.step()
            task_optimizer.step()
        # 4. Eval
        feature_extractor.eval()
        task_classifier.eval()
        with torch.no_grad():
            target_feature_eval = feature_extractor(target_X)
            pred_y_task_eval = task_classifier.predict(target_feature_eval)
            acc = sum(pred_y_task_eval == target_y_task) / target_y_task.shape[0]
            early_stopping(acc)
        loss_task_evals.append(acc.item())
        if early_stopping.early_stop & do_early_stop:
            break
        print(f"Epoch: {epoch}, Loss Domain: {loss_domain}, Loss Task: {loss_task}, Loss Pseudo Task: {loss_pseudo_task}, Acc: {acc}")
    # TODO: plot
    return feature_extractor, task_classifier, loss_task_evals


def _change_lr_during_jdot_training(
    feature_optimizer: torch.optim.Adam,
    task_optimizer: torch.optim.Adam,
    epoch: torch.Tensor,
    epoch_thr: int = 200,
    changed_lrs: List[float] = [0.00005, 0.00005],
):
    """
    Returns
    -------
    feature_optimizer : torch.optim.adam.Adam
    task_optimizer : torch.optim.adam.Adam
    """
    if epoch == epoch_thr:
        feature_optimizer.param_groups[0]["lr"] = changed_lrs[0]
        task_optimizer.param_groups[0]["lr"] = changed_lrs[0]
    return feature_optimizer, task_optimizer