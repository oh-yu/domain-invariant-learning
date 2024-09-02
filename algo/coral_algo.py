from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping, get_psuedo_label_weights


def get_MSE(x, y):
    return torch.sum((x - y) ** 2)


def get_covariance_matrix(x, y):
    N_x = x.shape[0]
    N_y = y.shape[0]
    average_x = torch.mean(x, dim=0)
    average_y = torch.mean(y, dim=0)
    cov_mat_x = (x - average_x).T @ (x - average_x) / (N_x - 1)
    cov_mat_y = (y - average_y).T @ (y - average_y) / (N_y - 1)
    return cov_mat_x, cov_mat_y


def fit(data, network, **kwargs):
    # Args
    source_loader, target_loader = data["source_loader"], data["target_loader"]
    target_X, target_y_task = data["target_X"], data["target_y_task"]

    feature_extractor = network["feature_extractor"]
    task_classifier = network["task_classifier"]
    task_optimizer = network["task_optimizer"]
    feature_optimizer = network["feature_optimizer"]

    config = {
        "num_epochs": 1000,
        "alpha": 1,
        "device": utils.DEVICE,
        "is_psuedo_weights": False,
        "is_changing_lr": False,
        "epoch_thr_for_changing_lr": 200,
        "changed_lrs": [0.00005, 0.00005],
        "stop_during_epochs": False,
        "epoch_thr_for_stopping": 2,
        "do_plot": False,
        "do_early_stop": False,
    }
    config.update(kwargs)
    num_epochs = config["num_epochs"]
    alpha = config["alpha"]
    device = config["device"]
    is_psuedo_weights = config["is_psuedo_weights"]
    is_changing_lr = config["is_changing_lr"]
    epoch_thr_for_changing_lr = config["epoch_thr_for_changing_lr"]
    changed_lrs = config["changed_lrs"]
    stop_during_epochs = config["stop_during_epochs"]
    epoch_thr_for_stopping = config["epoch_thr_for_stopping"]
    do_plot = config["do_plot"]
    do_early_stop = config["do_early_stop"]

    # Fit
    early_stopping = EarlyStopping()
    loss_corals = []
    loss_tasks = []
    loss_evals = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        task_classifier.train()
        feature_extractor.train()
        if stop_during_epochs & (epoch == epoch_thr_for_stopping):
            break
        if is_changing_lr:
            feature_optimizer, task_optimizer = _change_lr_during_coral_training(
                feature_optimizer, task_optimizer, epoch, epoch_thr=epoch_thr_for_changing_lr, changed_lrs=changed_lrs,
            )

        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(source_loader, target_loader):
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
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)
            source_out = task_classifier(source_X_batch)
            target_out = task_classifier(target_X_batch)

            # 1.1 Task Loss
            if task_classifier.output_size == 1:
                source_preds = torch.sigmoid(source_out).reshape(-1)
                criterion_weight = nn.BCELoss(weight=weights)
                loss_task = criterion_weight(source_preds, source_y_task_batch)
            else:
                source_preds = torch.softmax(source_out, dim=1)
                criterion_weight = nn.CrossEntropyLoss(reduction="none")
                loss_task = criterion_weight(source_preds, source_y_task_batch)
                loss_task = loss_task * weights
                loss_task = loss_task.mean()

            # 1.2 CoRAL Loss
            cov_mat_source, cov_mat_target = get_covariance_matrix(source_out, target_out)
            k = source_out.shape[1]
            loss_coral = get_MSE(cov_mat_source, cov_mat_target) * (1 / (4 * k ** 2))
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
            target_out = task_classifier.predict(feature_extractor(target_X))
            acc = sum(target_out == target_y_task) / len(target_y_task)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss Coral: {loss_coral}, Loss Task: {loss_task}, Acc: {acc}")
            early_stopping(acc.item())
            loss_evals.append(acc.item())
        if early_stopping.early_stop & do_early_stop:
            break

    if do_plot:
        plot_coral_loss(loss_corals, loss_tasks, loss_evals)
    return feature_extractor, task_classifier, None


def plot_coral_loss(loss_corals, loss_tasks, loss_evals):
    plt.plot(loss_corals, label="loss coral")
    plt.plot(loss_tasks, label="loss task")
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(loss_evals, label="eval acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()


def _change_lr_during_coral_training(
    feature_optimizer: torch.optim.Adam,
    task_optimizer: torch.optim.Adam,
    epoch: torch.Tensor,
    epoch_thr: int = 200,
    changed_lrs: List[float] = [0.00005],
):
    if epoch == epoch_thr:
        feature_optimizer.param_groups[0]["lr"] = changed_lrs[0]
        task_optimizer.param_groups[0]["lr"] = changed_lrs[0]
    return feature_optimizer, task_optimizer
