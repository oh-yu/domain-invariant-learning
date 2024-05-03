from sklearn.gaussian_process.kernels import RBF
import torch
from torch import nn, optim

from ..utils import utils
from ..networks import Decoder, Encoder

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

def fit_dan(source_loader, target_loader, num_epochs,
            feature_extractor, task_classifier_source, task_classifier_target,
            criterion, feature_optimizer, task_optimizer_source, task_optimizer_target,
            is_psuedo_weights, alpha_feat, alpha_task):

    for epoch in range(1, num_epochs+1):
        feature_extractor.train()
        task_classifier_source.train()
        task_classifier_target.train()


        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(
            source_loader, target_loader
        ):
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
        # TODO: Implement

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
    hidden_size = 10
    num_domains = 1
    num_classes = 1

    feature_extractor = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(utils.DEVICE)
    task_classifier_source = Decoder(input_size=hidden_size, output_size=num_classes, fc1_size=50, fc2_size=10).to(utils.DEVICE)
    task_classifier_target = Decoder(input_size=hidden_size, output_size=num_classes, fc1_size=50, fc2_size=10).to(utils.DEVICE)
    learning_rate = 0.001

    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    task_optimizer_target = optim.Adam(task_classifier_target.parameters(), lr=learning_rate)
    task_optimizer_source = optim.Adam(task_classifier_source.parameters(), lr=learning_rate)

    # Fit DAN
    fit_dan(
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
        alpha_feat=1,
        alpha_task=1,
    )
    print("Done!")

    # Eval