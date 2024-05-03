from sklearn.gaussian_process.kernels import RBF
import torch

from ..utils import utils


def fit_kernel(x, y):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html
    """
    rbf = RBF(length_scale=1.0)
    return rbf(x, y)

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
            criterion, optimizer, is_psuedo_weights, alpha_feat, alpha_task):

    for epoch in range(1, num_epochs.item() + 1):
        feature_extractor.train()
        task_classifier_source.train()
        task_classifier_target.train()


        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(
            source_loader, target_loader
        ):
            # 0. Data
            if task_classifier_source.output_size == 1:
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
            # 1.1 Feature Extractor
            source_feat = feature_extractor(source_X_batch)
            target_feat = feature_extractor(target_X_batch)

            # 1.2 Task Classifier
            source_out = task_classifier_source(source_feat)
            target_out = task_classifier_target(target_feat)

            # 1.3 Task Loss
            loss_task = criterion(source_out, source_y_task_batch)

            # 1.4 MMD Loss
            loss_mmd = get_MMD(source_feat, target_feat) * alpha_feat
            loss_mmd += get_MMD(source_out, target_out) * alpha_task
            loss = loss_task + loss_mmd

            # 2. Backward
            optimizer.zero_grad()
            loss.backward()

            # 3. Update Params
            optimizer.step()
        
        # 4. Eval
        # TODO: Implement