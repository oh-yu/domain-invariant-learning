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
    feature_extractor, task_classifier  = (
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
        for (source_X_batch, source_Y_batch), (target_X_batch, _), (target_prime_X_batch, _) in zip(source_loader, target_loader, target_prime_loader):
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
            criterion_pseudo = nn.CrossEntropyLoss(reduction='none')
            if task_classifier.output_size == 1:
                pred_target_y_task = torch.cat([(1-pred_target_y_task).reshape(-1, 1), pred_target_y_task.reshape(-1, 1)], dim=1)
            num_classes = pred_target_y_task.shape[1]
            pred_target_y_task = pred_target_y_task.unsqueeze(0).expand(len(source_y_task_batch), -1, -1)
            source_y_task_batch_expanded = source_y_task_batch.unsqueeze(1).expand(-1, len(pred_target_y_task))
            loss_pseudo_task_mat_dim1 = criterion_pseudo(pred_target_y_task.reshape(-1, num_classes), source_y_task_batch_expanded.reshape(-1).to(torch.long)).reshape(len(source_y_task_batch_expanded), len(pred_target_y_task)).T
            loss_pseudo_task_mat_dim1 = loss_pseudo_task_mat_dim1.to("cpu")
            cost_mat_dim1 = loss_domain_mat_dim1 + loss_pseudo_task_mat_dim1
            optimal_transport_weights_dim1 = ot.emd2(torch.ones(len(target_X_batch)) / len(target_X_batch), torch.ones(len(source_X_batch)) / len(source_X_batch), cost_mat_dim1)

            ## 1.3.2 Dim2
            loss_domain_mat_dim2 = torch.cdist(target_prime_X_batch, source_X_batch, p=2).to("cpu")
            if task_classifier.output_size == 1:
                pred_target_prime_y_task = torch.cat([(1-pred_target_prime_y_task).reshape(-1, 1), pred_target_prime_y_task.reshape(-1, 1)], dim=1)
            pred_target_prime_y_task = pred_target_prime_y_task.unsqueeze(0).expand(len(source_y_task_batch), -1, -1)
            source_y_task_batch_expanded = source_y_task_batch.unsqueeze(1).expand(-1, len(pred_target_prime_y_task))
            loss_pseudo_task_mat_dim2 = criterion_pseudo(pred_target_prime_y_task.reshape(-1, num_classes), source_y_task_batch_expanded.reshape(-1).to(torch.long)).reshape(len(source_y_task_batch_expanded), len(pred_target_prime_y_task)).T
            loss_pseudo_task_mat_dim2 = loss_pseudo_task_mat_dim2.to("cpu")
            cost_mat_dim2 = loss_domain_mat_dim2 + loss_pseudo_task_mat_dim2
            optimal_transport_weights_dim2 = ot.emd2(torch.ones(len(target_prime_X_batch)) / len(target_prime_X_batch), torch.ones(len(source_X_batch)) / len(source_X_batch), cost_mat_dim2)

            # 1.4 Align Loss
            loss_domain = torch.mean(optimal_transport_weights_dim1*loss_domain_mat_dim1)
            loss_domain += torch.mean(optimal_transport_weights_dim2*loss_domain_mat_dim2)
            
