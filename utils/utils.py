import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COL_IDX_TASK = 0
COL_IDX_DOMAIN = 1


def get_source_target_from_make_moons(n_samples=100, noise=0.05, rotation_degree=-30):
    # pylint: disable=too-many-locals
    # It seems reasonable in this case, since this method needs all of that.
    source_X, source_y = make_moons(n_samples=n_samples, noise=noise, random_state=1111)
    source_X[:, 0] -= 0.5

    theta = np.radians(rotation_degree)
    cos, sin = np.cos(theta), np.sin(theta)
    rotate_matrix = np.array([[cos, -sin], [sin, cos]])
    target_X = source_X.dot(rotate_matrix)
    target_y = source_y

    theta = np.radians(rotation_degree * 2)
    cos, sin = np.cos(theta), np.sin(theta)
    rotate_matrix = np.array([[cos, -sin], [sin, cos]])
    target_prime_X = source_X.dot(rotate_matrix)
    target_prime_y = source_y

    x1_min, x2_min = np.min([source_X.min(0), target_prime_X.min(0)], axis=0)
    x1_max, x2_max = np.max([source_X.max(0), target_prime_X.max(0)], axis=0)
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1_min - 0.1, x1_max + 0.1, 100), np.linspace(x2_min - 0.1, x2_max + 0.1, 100)
    )
    x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], axis=0)
    return source_X, source_y, target_X, target_y, target_prime_X, target_prime_y, x_grid, x1_grid, x2_grid


def get_loader(
    source_X: np.ndarray,
    target_X: np.ndarray,
    source_y_task: np.ndarray,
    target_y_task: np.ndarray,
    batch_size: int = 34,
    shuffle: bool = False,
    return_ds: bool = False,
):
    """
    Get instances of torch.utils.data.DataLoader for domain invariant learning,
    also return source and target data instantiated as torch.Tensor.

    Parameters
    ----------
    source_X : ndarray of shape(N, D) or (N, T, D)
    target_X : ndarray of shape(N, D) or (N, T, D)
    source_y_task : ndarray of shape(N, )
    target_y_task : ndarray of shape(N, )
    batch_size : int
    shuffle : boolean

    Returns
    -------
    source_loader : torch.utils.data.dataloader.DataLoader
        Contains source's feature, task label and domain label.
    target_loader : torch.utils.data.dataloader.DataLoader
        Contains target's feature, domain label.

    source_X : torch.Tensor of shape(N, D) or (N, T, D)
    source_y_task : ndarray of shape(N, 1)
    target_X : torch.Tensor of shape(N, D) or (N, T, D)
    target_y_task : torch.Tensor of shape(N, )
    """
    # 1. Create y_domain
    if source_y_task.ndim > 1:
        source_y_domain = np.zeros(source_y_task.shape[0]).reshape(-1, 1)
    else:
        source_y_domain = np.zeros_like(source_y_task).reshape(-1, 1)
        source_y_task = source_y_task.reshape(-1, 1)
    source_Y = np.concatenate([source_y_task, source_y_domain], axis=1)
    target_y_domain = np.ones_like(target_y_task)

    # 2. Instantiate torch.tensor
    # TODO: E1102: torch.tensor is not callable (not-callable)
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
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=shuffle)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=shuffle)

    if return_ds:
        return source_loader, target_loader, source_y_task, source_X, target_X, target_y_task, source_ds, target_ds
    else:
        return source_loader, target_loader, source_y_task, source_X, target_X, target_y_task


def apply_sliding_window(
    X: np.ndarray, y: np.ndarray, filter_len: int = 3, is_overlap: bool = True
) -> (np.ndarray, np.ndarray):
    """
    Parameters
    ----------
    X : ndarray of shape(N, H)
    y : ndarray of shape(N, )
    filter_len : int
    is_overlap: bool

    Returns
    -------
    filtered_X :
        ndarray of shape(N - filter_len + 1, filter_len, H) when is_ovelap == True:
        ndarray of shape(N//filter_len, filter_len, H) when is_ovelap == False:
    filtered_y :
        ndarray of shape(N - filter_len + 1, ) when is_ovelap == True:
        ndarray of shape(N//filter_len, ) when is_ovelap == False:
    """
    len_data, H = X.shape
    if is_overlap:
        N = len_data - filter_len + 1
        filtered_X = np.zeros((N, filter_len, H))
        for i in range(0, N):
            # print(f"(Start, End) = {i, i+filter_len-1}")
            start = i
            end = i + filter_len
            filtered_X[i] = X[start:end]
        filtered_y = y[filter_len - 1 :]
        return filtered_X, filtered_y

    else:
        X = np.expand_dims(X, axis=1)
        i = 0
        filtered_Xs = []
        filtered_ys = []
        while i < len_data - filter_len:
            filtered_X = np.expand_dims(np.concatenate(X[i : i + filter_len], axis=0), axis=0)
            filtered_Xs.append(filtered_X)
            filtered_ys.append(y[i + filter_len - 1])
            i += filter_len
        return np.vstack(filtered_Xs), np.array(filtered_ys).reshape(-1)


def fit_without_adaptation(source_loader, task_classifier, task_optimizer, criterion, num_epochs=1000, output_size=1):
    """
    Deep Learning model's fit method without domain adaptation.

    Parameters
    ----------
    source_loader : torch.utils.data.DataLoader
        Contains source's feature, task label and domain label.
        Domain Label is not used in this method.

    task_classifier : subclass of torch.nn.Module
        Target Deep Learning model.
        Currently it should output one dimensional prediction(only accept binary classification).

    task_optimizer : subclass of torch.optim.Optimizer
        Optimizer required instantiation with task_classifier.parameters().

    criterion : torch.nn.modules.loss.BCELoss
        Instance calculating Binary Cross Entropy Loss.

    num_epochs : int
    """
    for _ in range(num_epochs):
        for source_X_batch, source_Y_batch in source_loader:
            # Prep Data
            source_y_task_batch = source_Y_batch[:, COL_IDX_TASK]

            # Forward
            pred_y_task = task_classifier(source_X_batch)
            if output_size == 1:
                pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            else:
                source_y_task_batch = source_y_task_batch.to(torch.long)
                pred_y_task = torch.softmax(pred_y_task, dim=1)

            loss_task = criterion(pred_y_task, source_y_task_batch)

            # Backward
            task_optimizer.zero_grad()
            loss_task.backward()

            # Updata Params
            task_optimizer.step()
    return task_classifier


def visualize_tSNE(target_feature, source_feature):
    """
    Draw scatter plot including t-SNE encoded feature for source and target data.
    Small difference between them imply success of domain invarinat learning
    (only in the point of domain invariant).

    Parameters
    ----------
    target_feature : ndarray of shape(N, D)
        N is the number of samples, D is the number of features.

    source_feature : ndarray of shape(N, D)
    """
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=5)
    # TODO: Understand Argumetns for t-SNE
    N_target = target_feature.shape[0]
    feature = np.concatenate([target_feature, source_feature], axis=0)
    feature_tsne = tsne.fit_transform(feature)
    target_feature_tsne, source_feature_tsne = feature_tsne[:N_target], feature_tsne[N_target:]

    plt.figure()
    plt.scatter(source_feature_tsne[:, 0], source_feature_tsne[:, 1], label="Source", c="tan")
    plt.scatter(target_feature_tsne[:, 0], target_feature_tsne[:, 1], label="Target", c="lightsteelblue")
    plt.xlabel("tsne_X1")
    plt.ylabel("tsne_X2")
    plt.grid()
    # plt.legend()
    plt.show()

def visualize_tSNE_with_class_label(target_feature, source_feature, source_y_task, target_prime_y_task):
    """
    Draw scatter plot including t-SNE encoded feature for source and target data.
    Small difference between them imply success of domain invarinat learning
    (only in the point of domain invariant).

    Parameters
    ----------
    target_feature : ndarray of shape(N, D)
        N is the number of samples, D is the number of features.

    source_feature : ndarray of shape(N, D)
    """
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=5)
    # TODO: Understand Argumetns for t-SNE
    N_target = target_feature.shape[0]
    feature = np.concatenate([target_feature, source_feature], axis=0)
    feature_tsne = tsne.fit_transform(feature)
    target_feature_tsne, source_feature_tsne = feature_tsne[:N_target], feature_tsne[N_target:]

    plt.figure()
    plt.scatter(source_feature_tsne[:, 0], source_feature_tsne[:, 1], c=source_y_task)
    plt.scatter(target_feature_tsne[:, 0], target_feature_tsne[:, 1], c=target_prime_y_task)
    plt.xlabel("tsne_X1")
    plt.ylabel("tsne_X2")
    plt.grid()
    # plt.legend()
    plt.show()


def tensordataset_to_splitted_loaders(ds, batch_size):
    N_dataset = len(ds)
    train_idx = [i for i in range(0, N_dataset // 2, 1)]
    val_idx = [i for i in range(N_dataset // 2, N_dataset, 1)]
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader