from typing import List

import matplotlib.pyplot as plt
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