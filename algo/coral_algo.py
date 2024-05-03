from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from ..networks import Decoder
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


def fit_coral(source_loader, target_loader, num_epochs, task_classifier, criterion, optimizer, alpha, target_X, target_y_task):
    for epoch in range(1, num_epochs + 1):
        task_classifier.train()
        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(
            source_loader, target_loader
        ):
            # 0. Data
            source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK] > 0.5
            source_y_task_batch = source_y_task_batch.to(torch.float32)

            # 1. Forward
            source_out = task_classifier(source_X_batch)
            target_out = task_classifier(target_X_batch)

            # 1.1 Task Loss
            source_preds = torch.sigmoid(source_out).reshape(-1)
            loss_task = criterion(source_preds, source_y_task_batch)
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
        with torch.no_grad():
            task_classifier.eval()
            target_out = task_classifier(target_X)
            target_out = torch.sigmoid(target_out).reshape(-1)
            target_out = target_out > 0.5
            acc = sum(target_out == target_y_task) / len(target_y_task)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss Coral: {loss_coral}, Loss Task: {loss_task}, Acc: {acc}")
    return task_classifier


if __name__ == "__main__":
    # Load Data
    (
        source_X,
        target_X,
        source_y_task,
        target_y_task,
        x_grid,
        x1_grid,
        x2_grid,
    ) = utils.get_source_target_from_make_moons()
    source_loader, target_loader, source_y_task, source_X, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task
    )

    # Init NN
    num_classes = 1
    task_classifier = Decoder(input_size=2, output_size=num_classes, fc1_size=50, fc2_size=10).to(utils.DEVICE)
    learning_rate = 0.01

    criterion = nn.BCELoss()
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)

    # Fit CoRAL
    task_classifier = fit_coral(
        source_loader,
        target_loader,
        num_epochs=500,
        task_classifier=task_classifier,
        criterion=criterion,
        optimizer=task_optimizer,
        alpha=1,
        target_X=target_X,
        target_y_task=target_y_task,
    )
    source_X = source_X.cpu()
    target_X = target_X.cpu()
    x_grid = torch.tensor(x_grid, dtype=torch.float32).to(utils.DEVICE)
    y_grid = task_classifier(x_grid.T)
    y_grid = torch.sigmoid(y_grid).cpu().detach().numpy()

    plt.figure()
    plt.title("Domain Adaptation Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_X[:, 0], target_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()