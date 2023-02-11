import numpy as np
from sklearn.datasets import make_moons


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