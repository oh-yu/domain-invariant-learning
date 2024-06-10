import matplotlib.pyplot as plt
import torch
from sklearn.gaussian_process.kernels import RBF
from torch import nn, optim

from ..networks import Encoder, ThreeLayersDecoder
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


def fit(
    source_loader,
    target_loader,
    num_epochs,
    feature_extractor,
    task_classifier_source,
    task_classifier_target,
    criterion,
    feature_optimizer,
    task_optimizer_source,
    task_optimizer_target,
    is_psuedo_weights,
    alpha_feat,
    alpha_task,
    target_X,
    target_y_task,
):

    for epoch in range(1, num_epochs + 1):
        feature_extractor.train()
        task_classifier_source.train()
        task_classifier_target.train()

        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(source_loader, target_loader):
            # 0. Data
            source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK] > 0.5
            source_y_task_batch = source_y_task_batch.to(torch.float32)

            # 1. Forward
            # 1.1 Feature Extractor
            source_feat = feature_extractor(source_X_batch)
            target_feat = feature_extractor(target_X_batch)

            # 1.2 Task Classifier
            source_out = task_classifier_source(source_feat)
            target_out = task_classifier_target(target_feat)

            # 1.3 MMD Loss
            loss_mmd = get_MMD(source_feat, target_feat) * alpha_feat
            loss_mmd += get_MMD(source_out, target_out) * alpha_task

            # 1.4 Task Loss
            source_out = torch.sigmoid(source_out).reshape(-1)
            loss_task = criterion(source_out, source_y_task_batch)
            loss = loss_task + loss_mmd

            # 2. Backward
            feature_optimizer.zero_grad()
            task_optimizer_source.zero_grad()
            task_optimizer_target.zero_grad()
            loss.backward()

            # 3. Update Params
            feature_optimizer.step()
            task_optimizer_source.step()
            task_optimizer_target.step()

        # 4. Eval
        with torch.no_grad():
            feature_extractor.eval()
            task_classifier_target.eval()
            target_feat = feature_extractor(target_X)
            target_out = task_classifier_target(target_feat)
            target_out = torch.sigmoid(target_out).reshape(-1)
            target_out = target_out > 0.5
            acc = sum(target_out == target_y_task) / len(target_y_task)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss MMD: {loss_mmd}, Loss Task: {loss_task}, Acc: {acc}")
    return feature_extractor, task_classifier_target


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
    hidden_size = 100
    num_classes = 1

    feature_extractor = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(utils.DEVICE)
    task_classifier_source = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, fc1_size=500, fc2_size=100
    ).to(utils.DEVICE)
    task_classifier_target = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, fc1_size=500, fc2_size=100
    ).to(utils.DEVICE)
    learning_rate = 0.01

    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    task_optimizer_target = optim.Adam(task_classifier_target.parameters(), lr=learning_rate)
    task_optimizer_source = optim.Adam(task_classifier_source.parameters(), lr=learning_rate)

    # Fit DAN
    feature_extractor, task_classifier = fit(
        source_loader=source_loader,
        target_loader=target_loader,
        num_epochs=100,
        feature_extractor=feature_extractor,
        task_classifier_source=task_classifier_source,
        task_classifier_target=task_classifier_target,
        criterion=criterion,
        feature_optimizer=feature_optimizer,
        task_optimizer_source=task_optimizer_source,
        task_optimizer_target=task_optimizer_target,
        is_psuedo_weights=False,
        alpha_feat=0.5,
        alpha_task=0.1,
        target_X=target_X,
        target_y_task=target_y_task,
    )
    print("Done!")
    source_X = source_X.cpu()
    target_X = target_X.cpu()
    x_grid = torch.tensor(x_grid, dtype=torch.float32).to(utils.DEVICE)

    y_grid = task_classifier(feature_extractor(x_grid.T))
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

    # Without DA
    task_classifier = ThreeLayersDecoder(input_size=source_X.shape[1], output_size=num_classes).to(utils.DEVICE)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)
    task_classifier = utils.fit_without_adaptation(
        source_loader, task_classifier, task_optimizer, criterion, num_epochs=100
    )
    pred_y_task = task_classifier(target_X.to(utils.DEVICE))
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5
    acc = sum(pred_y_task == target_y_task) / target_y_task.shape[0]
    print(f"Without Adaptation Accuracy:{acc}")

    y_grid = task_classifier(x_grid.T)
    y_grid = torch.sigmoid(y_grid).cpu().detach().numpy()

    plt.figure()
    plt.title("Without Adaptation Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_X[:, 0], target_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()
