from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from .algo_utils import EarlyStopping, get_psuedo_label_weights, get_terminal_weights


class ReverseGradient(torch.autograd.Function):
    """
    https://arxiv.org/abs/1505.07818
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, step: torch.Tensor, num_steps: torch.Tensor):
        # TODO: Refactor num_steps, should not pass iteratively.
        ctx.save_for_backward(step, num_steps)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        step, num_steps, = ctx.saved_tensors
        scheduler = 2 / (1 + torch.exp(-10 * (step / (num_steps + 1)))) - 1
        # https://arxiv.org/pdf/1505.07818.pdf
        return grad_output * -1 * scheduler, None, None


def fit(data, network, **kwargs):
    """
    Fit Feature Extractor, Domain Classifier, Task Classifier by Domain Invarint Learning Algo.
    https://arxiv.org/abs/1505.07818

    Returns
    -------
    feature_extractor : subclass of torch.nn.Module
    task_classifier : subclass of torch.nn.Module
    loss_task_evals : list of float
    """
    # Args
    source_loader, target_loader = data["source_loader"], data["target_loader"]
    target_X, target_y_task = data["target_X"], data["target_y_task"]

    feature_extractor, domain_classifier, task_classifier = (
        network["feature_extractor"],
        network["domain_classifier"],
        network["task_classifier"],
    )
    criterion = network["criterion"]
    feature_optimizer, domain_optimizer, task_optimizer = (
        network["feature_optimizer"],
        network["domain_optimizer"],
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
    reverse_grad = ReverseGradient.apply
    early_stopping = EarlyStopping()

    # TODO: Understand torch.autograd.Function.apply
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
            domain_optimizer, feature_optimizer, task_optimizer = _change_lr_during_dann_training(
                domain_optimizer,
                feature_optimizer,
                task_optimizer,
                epoch,
                epoch_thr=epoch_thr_for_changing_lr,
                changed_lrs=changed_lrs,
            )

        for (source_X_batch, source_Y_batch), (target_X_batch, target_y_domain_batch) in zip(
            source_loader, target_loader
        ):
            # 0. Data
            if task_classifier.output_size == 1:
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK] > 0.5
                source_y_task_batch = source_y_task_batch.to(torch.float32)
                source_y_domain_batch = source_Y_batch[:, utils.COL_IDX_DOMAIN]
            else:
                if is_psuedo_weights:
                    output_size = source_Y_batch[:, :-1].shape[1]
                    source_y_task_batch = source_Y_batch[:, :output_size]
                    source_y_task_batch = torch.argmax(source_y_task_batch, dim=1)
                    source_y_task_batch = source_y_task_batch.to(torch.long)
                    source_y_domain_batch = source_Y_batch[:, output_size]
                else:
                    source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]
                    source_y_task_batch = source_y_task_batch.to(torch.long)
                    source_y_domain_batch = source_Y_batch[:, utils.COL_IDX_DOMAIN]

            psuedo_label_weights = get_psuedo_label_weights(source_Y_batch=source_Y_batch, device=device)

            # 1. Forward
            # 1.1 Feature Extractor
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)

            # 1.2. Domain Classifier
            source_X_batch_reversed_grad = reverse_grad(source_X_batch, epoch, num_epochs)
            target_X_batch = reverse_grad(target_X_batch, epoch, num_epochs)
            pred_source_y_domain = domain_classifier(source_X_batch_reversed_grad)
            pred_target_y_domain = domain_classifier(target_X_batch)
            pred_source_y_domain = torch.sigmoid(pred_source_y_domain).reshape(-1)
            pred_target_y_domain = torch.sigmoid(pred_target_y_domain).reshape(-1)

            loss_domain = criterion(pred_source_y_domain, source_y_domain_batch)
            loss_domain += criterion(pred_target_y_domain, target_y_domain_batch)
            loss_domains.append(loss_domain.item())

            # 1.3. Task Classifier
            pred_y_task = task_classifier.predict_proba(source_X_batch)
            weights = get_terminal_weights(
                is_target_weights,
                is_class_weights,
                is_psuedo_weights,
                pred_source_y_domain,
                source_y_task_batch,
                psuedo_label_weights,
            )
            if task_classifier.output_size == 1:
                criterion_weight = nn.BCELoss(weight=weights.detach())
                loss_task = criterion_weight(pred_y_task, source_y_task_batch)
                loss_tasks.append(loss_task.item())
            else:
                criterion_weight = nn.CrossEntropyLoss(reduction="none")
                loss_task = criterion_weight(pred_y_task, source_y_task_batch)
                loss_task = loss_task * weights
                loss_task = loss_task.mean()
                loss_tasks.append(loss_task.item())

            # 2. Backward, Update Params

            domain_optimizer.zero_grad()
            task_optimizer.zero_grad()
            feature_optimizer.zero_grad()

            loss_domain.backward(retain_graph=True)
            loss_task.backward()

            domain_optimizer.step()
            task_optimizer.step()
            feature_optimizer.step()

        # 3. Evaluation
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
        print(f"Epoch: {epoch}, Loss Domain: {loss_domain}, Loss Task: {loss_task}, Acc: {acc}")
    _plot_dann_loss(do_plot, loss_domains, loss_tasks, loss_task_evals)
    return feature_extractor, task_classifier, loss_task_evals


def _change_lr_during_dann_training(
    domain_optimizer: torch.optim.Adam,
    feature_optimizer: torch.optim.Adam,
    task_optimizer: torch.optim.Adam,
    epoch: torch.Tensor,
    epoch_thr: int = 200,
    changed_lrs: List[float] = [0.00005, 0.00005],
):
    """
    Returns
    -------
    domain_optimizer : torch.optim.adam.Adam
    feature_optimizer : torch.optim.adam.Adam
    task_optimizer : torch.optim.adam.Adam
    """
    if epoch == epoch_thr:
        domain_optimizer.param_groups[0]["lr"] = changed_lrs[1]
        feature_optimizer.param_groups[0]["lr"] = changed_lrs[0]
        task_optimizer.param_groups[0]["lr"] = changed_lrs[0]
    return domain_optimizer, feature_optimizer, task_optimizer


def _plot_dann_loss(
    do_plot: bool, loss_domains: List[float], loss_tasks: List[float], loss_task_evals: List[float]
) -> None:
    """
    plot domain&task losses for source, task loss for target.

    Parameters
    ----------
    do_plot: bool
    loss_domains: list of float
    loss_tasks: list of float
    loss_tasks_evals: list of float
    task loss for target data.
    """
    if do_plot:
        plt.figure()
        plt.plot(loss_domains, label="loss_domain")
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
