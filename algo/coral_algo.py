import torch
from torch import nn

from ..utils import utils
from .algo_utils import get_psuedo_label_weights, get_terminal_weights


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
    criterion = network["criterion"]
    task_optimizer = network["task_optimizer"]
    feature_optimizer = network["feature_optimizer"]

    config = {
        "num_epochs": 1000,
        "alpha": 1,
        "device": utils.DEVICE,
        "is_psuedo_weights": False
    }
    config.update(kwargs)
    num_epochs = config["num_epochs"]
    alpha = config["alpha"]
    device = config["device"]
    device = config["is_psuedo_weights"]

    # Fit
    for epoch in range(1, num_epochs + 1):
        task_classifier.train()
        feature_extractor.train()
        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(source_loader, target_loader):
            # 0. Data
            source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK] > 0.5
            source_y_task_batch = source_y_task_batch.to(torch.float32)

            if is_psuedo_weights:
                weights = get_psuedo_label_weights(source_Y_batch=source_Y_batch, device=device)
            else:
                weights = 1

            # 1. Forward
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)
            source_out = task_classifier(source_X_batch)
            target_out = task_classifier(target_X_batch)

            # 1.1 Task Loss
            source_preds = torch.sigmoid(source_out).reshape(-1)
            criterion_weight = nn.BCELoss(weight=weights.detach())
            loss_task = criterion_weight(source_preds, source_y_task_batch)

            # 1.2 CoRAL Loss
            cov_mat_source, cov_mat_target = get_covariance_matrix(source_out, target_out)
            k = source_out.shape[1]
            loss_coral = get_MSE(cov_mat_source, cov_mat_target) * (1 / (4 * k ** 2))
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
            target_out = task_classifier(feature_extractor(target_X))
            target_out = torch.sigmoid(target_out).reshape(-1)
            target_out = target_out > 0.5
            acc = sum(target_out == target_y_task) / len(target_y_task)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss Coral: {loss_coral}, Loss Task: {loss_task}, Acc: {acc}")
    return feature_extractor, task_classifier, None