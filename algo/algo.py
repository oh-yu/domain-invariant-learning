from typing import List

import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils


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


def fit(
    source_loader,
    target_loader,
    target_X,
    target_y_task,
    feature_extractor,
    domain_classifier,
    task_classifier,
    criterion,
    feature_optimizer,
    domain_optimizer,
    task_optimizer,
    num_epochs=1000,
    is_target_weights=False,
    is_class_weights=False,
    is_psuedo_weights=False,
    do_plot=False,
    do_print=False,
    device=utils.DEVICE,
    is_changing_lr=False,
    epoch_thr_for_changing_lr=200,
    changed_lrs=[0.00005, 0.00005],
    stop_during_epochs=False,
    epoch_thr_for_stopping=2,
):
    # pylint: disable=too-many-arguments, too-many-locals
    # It seems reasonable in this case, since this method needs all of that.
    """
    Fit Feature Extractor, Domain Classifier, Task Classifier by Domain Invarint Learning Algo.
    https://arxiv.org/abs/1505.07818

    Parameters
    ----------
    source_loader : torch.utils.data.dataloader.DataLoader
        Iterable containing batched source's feature, task label and domain label.

    target_loader : torch.utils.data.dataloader.DataLoader
        Iterable containing batched target's feature, domain label.

    target_X : torch.Tensor of shape(N, D) or (N, T, D)
        Sent to on GPU.

    target_y_task : torch.Tensor of shape(N, )
        Sent to on GPU.

    feature_extractor : subclass of torch.nn.Module
    domain_classifier : subclass of torch.nn.Module
    task_classifier : subclass of torch.nn.Module
    criterion : torch.nn.modules.loss.BCELoss
    feature_optimizer : subclass of torch.optim.Optimizer
    domain_optimizer : subclass of torch.optim.Optimizer
    task_optimizer : subclass of torch.optim.Optimizer
    num_epochs : int
    is_target_weights: bool
    is_class_weights: bool
    is_psuedo_weights: bool
    do_plot: bool

    Returns
    -------
    feature_extractor : subclass of torch.nn.Module
    task_classifier : subclass of torch.nn.Module
    loss_task_evals : list of float
    """

    reverse_grad = ReverseGradient.apply
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
                domain_optimizer, feature_optimizer, task_optimizer, epoch, epoch_thr=epoch_thr_for_changing_lr, changed_lrs=changed_lrs
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

            psuedo_label_weights = _get_psuedo_label_weights(source_Y_batch=source_Y_batch, device=device)

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
            weights = _get_terminal_weights(
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
        loss_task_evals.append(acc.item())

        print(f"Epoch: {epoch}, Loss Domain: {loss_domain}, Loss Task: {loss_task}, Acc: {acc}")
    utils._plot_dann_loss(do_plot, loss_domains, loss_tasks, loss_task_evals)
    return feature_extractor, task_classifier, loss_task_evals


def _get_psuedo_label_weights(source_Y_batch: torch.Tensor, thr: float = 0.75, alpha: int = 1, device=utils.DEVICE) -> torch.Tensor:
    """
    # TODO: attach paper

    Parameters
    ----------
    source_Y_batch : torch.Tensor of shape(N, 2)
    thr : float

    Returns
    -------
    psuedo_label_weights : torch.Tensor of shape(N, )
    """
    output_size = source_Y_batch[:, :-1].shape[1]
    psuedo_label_weights = []

    if output_size == 1:
        pred_y = source_Y_batch[:, utils.COL_IDX_TASK]        
        for i in pred_y:
            if i > thr:
                psuedo_label_weights.append(1)
            elif i < 1 - thr:
                psuedo_label_weights.append(1)
            else:
                if i > 0.5:
                    psuedo_label_weights.append(i**alpha + (1 - thr))
                else:
                    psuedo_label_weights.append((1 - i)**alpha + (1 - thr))

    else:
        pred_y = source_Y_batch[:, :output_size]
        pred_y = torch.max(pred_y, axis=1).values
        for i in pred_y:
            if i > thr:
                psuedo_label_weights.append(1)
            else:
                psuedo_label_weights.append(i**alpha + (1 - thr))
    return torch.tensor(psuedo_label_weights, dtype=torch.float32).to(device)


def _get_terminal_weights(
    is_target_weights: bool,
    is_class_weights: bool,
    is_psuedo_weights: bool,
    pred_source_y_domain: torch.Tensor,
    source_y_task_batch: torch.Tensor,
    psuedo_label_weights: torch.Tensor,
) -> torch.Tensor:
    """
    # TODO: attach paper

    Parameters
    ----------
    is_target_weights: bool
    is_class_weights: bool
    is_psuedo_weights: bool
    pred_source_y_domain : torch.Tensor of shape(N, )
    source_y_task_batch : torch.Tensor of shape(N, )
    psuedo_label_weights : torch.Tensor of shape(N, )

    Returns
    -------
    weights : torch.Tensor of shape(N, )
    terminal sample weights for nn.BCELoss
    """
    if is_target_weights:
        target_weights = pred_source_y_domain / (1 - pred_source_y_domain)
    else:
        target_weights = 1
    if is_class_weights:
        class_weights = _get_class_weights(source_y_task_batch)
    else:
        class_weights = 1
    if is_psuedo_weights:
        weights = target_weights * class_weights * psuedo_label_weights
    else:
        weights = target_weights * class_weights
    return weights


def _get_class_weights(source_y_task_batch):
    p_occupied = sum(source_y_task_batch) / source_y_task_batch.shape[0]
    p_unoccupied = 1 - p_occupied
    class_weights = torch.zeros_like(source_y_task_batch)
    for i, y in enumerate(source_y_task_batch):
        if y == 1:
            class_weights[i] = p_unoccupied
        elif y == 0:
            class_weights[i] = p_occupied
    return class_weights

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