import torch

from ..utils import utils

def get_psuedo_label_weights(
    source_Y_batch: torch.Tensor, thr: float = 0.75, alpha: int = 1, device=utils.DEVICE
) -> torch.Tensor:
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
                    psuedo_label_weights.append(i ** alpha + (1 - thr))
                else:
                    psuedo_label_weights.append((1 - i) ** alpha + (1 - thr))

    else:
        pred_y = source_Y_batch[:, :output_size]
        pred_y = torch.max(pred_y, axis=1).values
        for i in pred_y:
            if i > thr:
                psuedo_label_weights.append(1)
            else:
                psuedo_label_weights.append(i ** alpha + (1 - thr))
    return torch.tensor(psuedo_label_weights, dtype=torch.float32).to(device)


def get_terminal_weights(
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