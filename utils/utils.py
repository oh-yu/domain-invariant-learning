import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.manifold import TSNE
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COL_IDX_TASK = 0
COL_IDX_DOMAIN = 1


def get_source_target_from_make_moons(n_samples=100, noise=0.05, rotation_degree=-30):
    # pylint: disable=too-many-locals
    # It seems reasonable in this case, since this method needs all of that.
    """
    Get source and target data in domain adaptation problem,
    generated by sklean.datasets.make_moons.

    Parameters
    ----------
    n_samples : int
        Represents the number of make_moons samples to be generated.

    noise : int
        Represents standard deviation of gaussian noise.

    rotatin_degree : int
        Represents degree to be rotated.
        Used for generating unsupervised target data in domain adaptation problem.

    Returns
    -------
    source_X : ndarray of shape(n_samples, 2)
        The generated source feature.

    target_X : ndarray of shape(n_samples, 2)
        The generated target feature.

    source_y : ndarray of shape(n_samples, )
        The generated source label.

    target_y : ndarray of shape(n_samples, )
        The generated target label, this is not used for ML model training.

    x_grid : ndarray of shape(2, 10000)
        Stacked meshgrid points, each row is each dimension, used for visualization.

    x1_grid : ndarray of shape(100, 100)
        The first dimentional Meshgrid points, used for visualization.

    x2_grid : ndarray of shape(100, 100)
        The second dimentional Meshgrid points, used for visualization.
    """
    source_X, source_y = make_moons(n_samples=n_samples, noise=noise)
    source_X[:, 0] -= 0.5
    theta = np.radians(rotation_degree)
    cos, sin = np.cos(theta), np.sin(theta)
    rotate_matrix = np.array([[cos, -sin],[sin, cos]])
    target_X = source_X.dot(rotate_matrix)
    target_y = source_y

    x1_min, x2_min = np.min([source_X.min(0), target_X.min(0)], axis=0)
    x1_max, x2_max = np.max([source_X.max(0), target_X.max(0)], axis=0)
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min-0.1, x1_max+0.1, 100),
                                   np.linspace(x2_min-0.1, x2_max+0.1, 100))
    x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)], axis=0)
    return source_X, target_X, source_y, target_y, x_grid, x1_grid, x2_grid


def get_loader(source_X: np.ndarray, target_X: np.ndarray,
               source_y_task: np.ndarray, target_y_task: np.ndarray,
               batch_size: int = 34, shuffle: bool = False):
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

    return source_loader, target_loader, source_y_task, source_X, target_X, target_y_task


def apply_sliding_window(X: np.ndarray, y: np.ndarray, filter_len: int = 3) -> (np.ndarray, np.ndarray):
    """
    Parameters
    ----------
    X : ndarray of shape(N, H)
    y : ndarray of shape(N, )
    filter_len : int

    Returns
    -------
    filtered_X : ndarray of shape(N', filter_len, H)
    N' is N - filter_len + 1.
    filtered_y : ndarray of shape(N', )
    """
    len_data, H = X.shape
    N = len_data - filter_len + 1
    filtered_X = np.zeros((N, filter_len, H))
    for i in range(0, N):
        # print(f"(Start, End) = {i, i+filter_len-1}")
        start = i
        end = i+filter_len
        filtered_X[i] = X[start:end]
    filtered_y = y[filter_len-1:]
    return filtered_X, filtered_y


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc1(x))


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, fc1_size=50, fc2_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ManyToOneRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x):
        _, (x, _) = self.rnn(x)
        x = x[-1, :, :]
        return x


class Conv1d(nn.Module):
    # TODO: Understand nn.Conv1d doumentation
    def __init__(self, input_size: int, out_channels1: int = 128, out_channels2: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels1)
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(out_channels2)
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = torch.mean(x, dim=2)
        return x


class ReverseGradient(torch.autograd.Function):
    # https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

    @staticmethod
    def forward(ctx, x: torch.Tensor, step: torch.Tensor, num_steps: torch.Tensor):
    # TODO: Refactor num_steps, should not pass iteratively.
    # pylint: disable=arguments-differ
    # It seems better in this case, since this method need only one arg.
        ctx.save_for_backward(step, num_steps)
        return x
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
    # pylint: disable=arguments-differ
    # It seems better in this case, since this method need only one arg.
        step, num_steps, = ctx.saved_tensors
        scheduler = 2 / (1 + torch.exp(-10 * (step/(num_steps+1)))) - 1
        # https://arxiv.org/pdf/1505.07818.pdf

        # TODO: Check speficiation about correspondence between input for forward and return for backward.
        return grad_output * -1 * scheduler, None, None


def _change_lr_during_dann_training(domain_optimizer: torch.optim.Adam, feature_optimizer: torch.optim.Adam, task_optimizer: torch.optim.Adam,
                                    epoch: torch.Tensor, epoch_thr: int = 200, lr: float = 0.00005):
    """
    Returns
    -------
    domain_optimizer : torch.optim.adam.Adam
    feature_optimizer : torch.optim.adam.Adam
    task_optimizer : torch.optim.adam.Adam
    """
    if epoch == epoch_thr:
        domain_optimizer.param_groups[0]["lr"] = lr
        feature_optimizer.param_groups[0]["lr"] = lr
        task_optimizer.param_groups[0]["lr"] = lr
    return domain_optimizer, feature_optimizer, task_optimizer


def _get_psuedo_label_weights(source_Y_batch: torch.Tensor, thr: float = 0.75) -> torch.Tensor:
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
    pred_y = source_Y_batch[:, COL_IDX_TASK]
    psuedo_label_weights = []
    for i in pred_y:
        if i > thr:
            psuedo_label_weights.append(1)
        elif i < 1-thr:
            psuedo_label_weights.append(1)
        else:
            if i > 0.5:
                psuedo_label_weights.append(i + (1-thr))
            else:
                psuedo_label_weights.append((1-i) + (1-thr))
    psuedo_label_weights = torch.tensor(psuedo_label_weights, dtype=torch.float32).to(DEVICE)
    return psuedo_label_weights


def _get_terminal_weights(is_target_weights: bool, is_class_weights: bool, is_psuedo_weights: bool,
                          pred_source_y_domain: torch.Tensor, source_y_task_batch: torch.Tensor, psuedo_label_weights: torch.Tensor) -> torch.Tensor:
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
        target_weights = pred_source_y_domain / (1-pred_source_y_domain)
    else:
        target_weights = 1
    if is_class_weights:
        class_weights = get_class_weights(source_y_task_batch)
    else:
        class_weights = 1
    if is_psuedo_weights:
        weights = target_weights * class_weights * psuedo_label_weights
    else:
        weights = target_weights * class_weights
    return weights


def _plot_dann_loss(do_plot: bool, loss_domains: List[float], loss_tasks: List[float], loss_task_evals: List[float]) -> None:
    """
    plot domain&task losses for source, task loss for target.

    Parameters
    ----------
    do_plot: bool
    loss_domains: list of float
    loss_tasks: list of float
    loss_tasks_evals: list of float
    task loss for target data.
    """
    if do_plot:
        plt.figure()
        plt.plot(loss_domains, label="loss_domain")
        plt.plot(loss_tasks, label="loss_task")
        plt.xlabel("batch")
        plt.ylabel("binary cross entropy loss")
        plt.legend()

        plt.figure()
        plt.plot(loss_task_evals, label="loss_task_eval")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()


def fit(source_loader, target_loader, target_X, target_y_task,
        feature_extractor, domain_classifier, task_classifier, criterion,
        feature_optimizer, domain_optimizer, task_optimizer, num_epochs=1000,
        is_target_weights=True, is_class_weights=False, is_psuedo_weights=False,
        do_plot=False):
    # pylint: disable=too-many-arguments, too-many-locals
    # It seems reasonable in this case, since this method needs all of that.
    """
    Fit Feature Extractor, Domain Classifier, Task Classifier by Domain Invarint Learning Algo.

    Parameters
    ----------
    source_loader : torch.utils.data.dataloader.DataLoader
        Iterable containing batched source's feature, task label and domain label.

    target_loader : torch.utils.data.dataloader.DataLoader
        Iterable containing batched target's feature, domain label.

    target_X : torch.Tensor of shape(N, D) or (N, T, D)
        Sent to on GPU.

    target_y_task : torch.Tensor of shape(N, )
        Sent to on GPU.

    feature_extractor : subclass of torch.nn.Module
    domain_classifier : subclass of torch.nn.Module
    task_classifier : subclass of torch.nn.Module
    criterion : torch.nn.modules.loss.BCELoss
    feature_optimizer : subclass of torch.optim.Optimizer
    domain_optimizer : subclass of torch.optim.Optimizer
    task_optimizer : subclass of torch.optim.Optimizer
    num_epochs : int
    is_target_weights: bool
    is_class_weights: bool
    is_psuedo_weights: bool
    do_plot: bool

    Returns
    -------
    feature_extractor : subclass of torch.nn.Module
    task_classifier : subclass of torch.nn.Module
    loss_task_evals : list of float
    """

    reverse_grad = ReverseGradient.apply
    # TODO: Understand torch.autograd.Function.apply
    loss_domains = []
    loss_tasks = []
    loss_task_evals = []
    num_epochs = torch.tensor(num_epochs, dtype=torch.int32).to(DEVICE)

    for epoch in range(1, num_epochs.item()+1):
        epoch = torch.tensor(epoch, dtype=torch.float32).to(DEVICE)
        feature_extractor.train()
        task_classifier.train()
        domain_optimizer, feature_optimizer, task_optimizer = _change_lr_during_dann_training(domain_optimizer, feature_optimizer, task_optimizer, epoch)

        for (source_X_batch, source_Y_batch), (target_X_batch, target_y_domain_batch) in zip(source_loader, target_loader):
            # 0. Data
            source_y_task_batch = source_Y_batch[:, COL_IDX_TASK] > 0.5
            source_y_task_batch = source_y_task_batch.to(torch.float32)
            psuedo_label_weights = _get_psuedo_label_weights(source_Y_batch)
            source_y_domain_batch = source_Y_batch[:, COL_IDX_DOMAIN]

            # 1. Forward
            # 1.1 Feature Extractor
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)

            # 1.2. Domain Classifier
            source_X_batch_reversed_grad = reverse_grad(source_X_batch, epoch, num_epochs)
            target_X_batch = reverse_grad(target_X_batch, epoch, num_epochs)
            pred_source_y_domain = domain_classifier(source_X_batch_reversed_grad)
            pred_target_y_domain = domain_classifier(target_X_batch)
            pred_source_y_domain = torch.sigmoid(pred_source_y_domain).reshape(-1)
            pred_target_y_domain = torch.sigmoid(pred_target_y_domain).reshape(-1)

            loss_domain = criterion(pred_source_y_domain, source_y_domain_batch)
            loss_domain += criterion(pred_target_y_domain, target_y_domain_batch)
            loss_domains.append(loss_domain.item())

            # 1.3. Task Classifier
            pred_y_task = task_classifier(source_X_batch)
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            weights = _get_terminal_weights(is_target_weights, is_class_weights, is_psuedo_weights,
                                            pred_source_y_domain, source_y_task_batch, psuedo_label_weights)
            criterion_weight = nn.BCELoss(weight=weights.detach())
            loss_task = criterion_weight(pred_y_task, source_y_task_batch)
            loss_tasks.append(loss_task.item())

            # 2. Backward, Update Params
            domain_optimizer.zero_grad()
            task_optimizer.zero_grad()
            feature_optimizer.zero_grad()

            loss_domain.backward(retain_graph = True)
            loss_task.backward()

            domain_optimizer.step()
            task_optimizer.step()
            feature_optimizer.step()

        # 3. Evaluation
        feature_extractor.eval()
        task_classifier.eval()
        with torch.no_grad():
            target_feature_eval = feature_extractor(target_X)
            pred_y_task_eval = task_classifier(target_feature_eval)
            pred_y_task_eval = torch.sigmoid(pred_y_task_eval).reshape(-1)
            pred_y_task_eval = pred_y_task_eval > 0.5
            acc = sum(pred_y_task_eval == target_y_task) / target_y_task.shape[0]
        loss_task_evals.append(acc.item())
    _plot_dann_loss(do_plot, loss_domains, loss_tasks, loss_task_evals)
    return feature_extractor, task_classifier, loss_task_evals


def fit_without_adaptation(source_loader, task_classifier,
                           task_optimizer, criterion, num_epochs=1000):
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
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
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
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=5)
    # TODO: Understand Argumetns for t-SNE
    target_feature_tsne = tsne.fit_transform(target_feature)
    source_feature_tsne = tsne.fit_transform(source_feature)

    plt.figure()
    plt.scatter(source_feature_tsne[:, 0], source_feature_tsne[:, 1], label="Source")
    plt.scatter(target_feature_tsne[:, 0], target_feature_tsne[:, 1], label="Target")
    plt.xlabel("tsne_X1")
    plt.ylabel("tsne_X2")
    plt.legend()
    plt.show()


def get_class_weights(source_y_task_batch):
    p_occupied = sum(source_y_task_batch) / source_y_task_batch.shape[0] 
    p_unoccupied = 1 - p_occupied
    class_weights = torch.zeros_like(source_y_task_batch)
    for i, y in enumerate(source_y_task_batch):
        if y == 1:
            class_weights[i] = p_unoccupied
        elif y == 0:
            class_weights[i] = p_occupied
    return class_weights
