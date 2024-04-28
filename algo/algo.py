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
    is_target_weights=True,
    is_class_weights=False,
    is_psuedo_weights=False,
    do_plot=False,
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
    num_epochs = torch.tensor(num_epochs, dtype=torch.int32).to(utils.DEVICE)
    
    for epoch in tqdm(range(1, num_epochs.item() + 1)):
        epoch = torch.tensor(epoch, dtype=torch.float32).to(utils.DEVICE)
        feature_extractor.train()
        task_classifier.train()
        domain_optimizer, feature_optimizer, task_optimizer = utils._change_lr_during_dann_training(
            domain_optimizer, feature_optimizer, task_optimizer, epoch
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

            psuedo_label_weights = utils._get_psuedo_label_weights(source_Y_batch)

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
            weights = utils._get_terminal_weights(
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
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Loss Domain: {loss_domain}, Loss Task: {loss_task}, Acc: {acc}")
    utils._plot_dann_loss(do_plot, loss_domains, loss_tasks, loss_task_evals)
    return feature_extractor, task_classifier, loss_task_evals
