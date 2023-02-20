import numpy as np
from sklearn.datasets import make_moons
import torch
from torch.utils.data import TensorDataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_source_target():
    Xs, ys = make_moons(n_samples=100, noise=0.05)
    Xs[:, 0] -= 0.5
    theta = np.radians(-30)
    cos, sin = np.cos(theta), np.sin(theta)
    rotate_matrix = np.array([[cos, -sin],[sin, cos]])
    Xs_rotated = Xs.dot(rotate_matrix)
    ys_rotated = ys

    x1_min, x2_min = np.min([Xs.min(0), Xs_rotated.min(0)], 0)
    x1_max, x2_max = np.max([Xs.max(0), Xs_rotated.max(0)], 0)
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min-0.1, x1_max+0.1, 100), np.linspace(x2_min-0.1, x2_max+0.1, 100))
    x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)])
    return Xs, Xs_rotated, ys, ys_rotated, x_grid, x1_grid, x2_grid


def get_loader(source_X, target_X, source_y_task, target_y_task):
    # 1. Create y_domain
    source_y_domain = np.zeros_like(source_y_task).reshape(-1, 1)
    source_y_task = source_y_task.reshape(-1, 1)
    source_Y = np.concatenate([source_y_task, source_y_domain], axis=1)
    target_y_domain = np.ones_like(target_y_task)

    # 2. Instantiate torch.tensor
    source_X = torch.tensor(source_X, dtype=torch.float32)
    source_Y = torch.tensor(source_Y, dtype=torch.float32)
    target_X = torch.tensor(target_X, dtype=torch.float32)
    target_y_domain = torch.tensor(target_y_domain, dtype=torch.float32)
    target_y_task = torch.tensor(target_y_task, dtype=torch.float32)
    
    # 3. To GPU
    source_X = source_X.to(DEVICE)
    source_Y = source_Y.to(DEVICE)
    target_X = target_X.to(DEVICE)
    target_y_domain = target_y_domain.to(DEVICE)
    target_y_task = target_y_task.to(DEVICE)
    
    # 4. Instantiate DataLoader
    source_ds = TensorDataset(source_X, source_Y)
    target_ds = TensorDataset(target_X, target_y_domain)
    source_loader = DataLoader(source_ds, batch_size=34, shuffle=True)
    target_loader = DataLoader(target_ds, batch_size=34, shuffle=True)
    
    return source_loader, target_loader, source_y_task, source_X, target_X, target_y_task