from sklearn.gaussian_process.kernels import RBF
import numpy as np
import torch
from ..utils import utils


def fit_kernel(x, y):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html
    """
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    rbf = RBF(length_scale=1.0)
    return torch.tensor(rbf(x, y), dtype=torch.float32).to(utils.DEVICE)

def get_MMD(x, y):
    """
    https://jejjohnson.github.io/research_journal/appendix/similarity/mmd/
    """
    mmd_xx = torch.mean(fit_kernel(x, x))
    mmd_yy = torch.mean(fit_kernel(y, y))
    mmd_xy = -2 * torch.mean(fit_kernel(x, y))
    return mmd_xx + mmd_yy + mmd_xy


def get_covariance_matrix(x, y):
    N_x = x.shape[0]
    N_y = y.shape[0]
    average_x = torch.mean(x, dim=0)
    average_y = torch.mean(y, dim=0)
    cov_mat_x = (x - average_x).T @ (x - average_x) / (N_x - 1)
    cov_mat_y = (y - average_y).T @ (y - average_y) / (N_y - 1)
    return cov_mat_x, cov_mat_y


def fit_coral(source_loader, target_loader, num_epochs, task_classifier, criterion, optimizer, is_psuedo_weights, k, alpha):
    for epoch in range(1, num_epochs.item() + 1):
        task_classifier.train()
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
            # 1. Forward
            source_out = task_classifier(source_X_batch)
            target_out = task_classifier(target_X_batch)

            # 1.1 Task Loss
            loss_task = criterion(source_out, source_y_task_batch)
            # 1.2 CoRAL Loss
            cov_mat_source, cov_mat_target = get_covariance_matrix(source_out, target_out)
            k = source_out.shape[1]
            loss_coral = get_MMD(cov_mat_source, cov_mat_target) * (1/(4*k**2))
            loss = loss_task + loss_coral * alpha
            # 2. Backward
            optimizer.zero_grad()
            loss.backward()
            # 3. Update Params
            optimizer.step()

        # 4. Eval
        # TODO: Implement