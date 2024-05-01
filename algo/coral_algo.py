import torch
from ..utils import utils

def fit_coral(source_loader, target_loader, num_epochs, task_classifier, criterion, optimizer, is_psuedo_weights):
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
            # 1.1 Task Loss
            # 1.2 CoRAL Loss

            # 2. Backward
            # 3. Update Params
            pass
        # 4. Eval