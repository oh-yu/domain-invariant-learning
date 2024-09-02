from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping
from .coral_algo import get_covariance_matrix, get_MSE

def fit(data, network, **kwargs):
    # Args
    source_loader, target_loader, target_prime_loader = (
        data["source_loader"],
        data["target_loader"],
        data["target_prime_loader"],
    )
    target_prime_X, target_prime_y_task = data["target_prime_X"], data["target_prime_y_task"]
    feature_extractor, task_classifier_dim1, task_classifier_dim2  = (
        network["feature_extractor"],
        network["task_classifier_dim1"],
        network["task_classifier_dim2"],
    )
    criterion = network["criterion"]
    feature_optimizer, task_optimizer_dim1, task_optimizer_dim2 = (
        network["feature_optimizer"],
        network["task_optimizer_dim1"],
        network["task_optimizer_dim2"],
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
    loss_tasks = []
    loss_task_evals = []
    loss_corals = []
    early_stopping = EarlyStopping()
    num_epochs = torch.tensor(num_epochs, dtype=torch.int32).to(device)
    for epoch in tqdm(range(1, num_epochs + 1)):
        task_classifier_dim1.train()
        task_classifier_dim2.train()
        feature_extractor.train()

        for (source_X_batch, source_Y_batch), (target_X_batch, _), (target_prime_X_batch, ) in zip(source_loader, target_loader, target_prime_loader):
            pass
        