from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping
from .coral_algo import get_covariance_matrix, get_MSE, plot_coral_loss

def fit(data, network, **kwargs):
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
        "alpha": 1,
        "num_epochs": 1000,
        "device": utils.DEVICE,
        "do_early_stop": False,
        "do_plot": False,
    }
    config.update(kwargs)
    alpha = config["alpha"]
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
        task_classifier.train()
        feature_extractor.train()

        for (source_X_batch, source_Y_batch), (target_X_batch, _), (target_prime_X_batch, _) in zip(source_loader, target_loader, target_prime_loader):
            if task_classifier.output_size == 1:
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]
                source_y_task_batch = source_y_task_batch.to(torch.float32)
            else:
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]
                source_y_task_batch = source_y_task_batch.to(torch.long)
            
            # 1. Forward
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)
            target_prime_X_batch = feature_extractor(target_prime_X_batch)
            source_out = task_classifier(source_X_batch)
            target_out = task_classifier(target_X_batch)
            target_prime_out = task_classifier(target_prime_X_batch)
            ## 1.1 Task Loss
            if task_classifier.output_size == 1:
                source_preds = torch.sigmoid(source_out).reshape(-1)
                loss_task = criterion(source_preds, source_y_task_batch)
            else:
                criterion = nn.CrossEntropyLoss()
                source_preds = torch.softmax(source_out, dim=1)
                loss_task = criterion(source_preds, source_y_task_batch)
                loss_task = loss_task.mean()

            ## 1.2 CoRAL Loss
            cov_mat_source, cov_mat_target = get_covariance_matrix(source_out, target_out)
            _, cov_mat_target_prime = get_covariance_matrix(source_out, target_prime_out)

            k = source_out.shape[1]
            loss_coral = get_MSE(cov_mat_source, cov_mat_target) * (1 / (4 * k ** 2))
            loss_coral += get_MSE(cov_mat_source, cov_mat_target_prime) * (1 / (4 * k ** 2))

            loss_corals.append(loss_coral.item())
            loss_tasks.append(loss_task.item())
            loss = loss_task + loss_coral * alpha

            # 2. Backward
            task_optimizer.zero_grad()
            feature_optimizer.zero_grad()
            loss.backward()

            # 3. Update Params
            task_optimizer.step()
            feature_optimizer.step()
        # 4. Eval
        with torch.no_grad():
            feature_extractor.eval()
            task_classifier.eval()
            target_prime_out = task_classifier.predict(feature_extractor(target_prime_X))
            acc = sum(target_prime_out == target_prime_y_task) / len(target_prime_y_task)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss Coral: {loss_coral}, Loss Task: {loss_task}, Acc: {acc}")
            early_stopping(acc.item())
            loss_task_evals.append(acc.item())
        if early_stopping.early_stop & do_early_stop:
            break
    if do_plot:
        plot_coral_loss(loss_corals, loss_tasks, loss_task_evals)
    return feature_extractor, task_classifier, None