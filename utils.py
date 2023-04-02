import matplotlib.pyplot as plt
import numpy as np
from ray import tune
from sklearn.datasets import make_moons
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_source_target_from_make_moons(n_samples=100, noise=0.05, rotation_degree=-30):
    """
    Get a svm instance with the best parameter, trained by GridSearchCV().
    Current params were selected by evaluation on a wider grid and looking at the heatmap.

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
    X : ndarray of shape(n_samples, 2)
        The generated source feature.
    X_rotated : ndarray of shape(n_samples, 2)
        The generated target feature.
    y : ndarray of shape(n_samples, )
        The generated source label.
    y_rotated : ndarray of shape(n_samples, )
        The generated target label, this is not used for training.
    x_grid : ndarray of shape(2, 10000)
        Stacked meshgrid points, each row is each dimension.
    x1_grid : ndarray of shape(100, 100)
        Feature's first dimentional Meshgrid points.
    x2_grid : ndarray of shape(100, 100)
        Feature's second dimentional Meshgrid points.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise)
    X[:, 0] -= 0.5
    theta = np.radians(rotation_degree)
    cos, sin = np.cos(theta), np.sin(theta)
    rotate_matrix = np.array([[cos, -sin],[sin, cos]])
    X_rotated = X.dot(rotate_matrix)
    y_rotated = y

    x1_min, x2_min = np.min([X.min(0), X_rotated.min(0)], 0)
    x1_max, x2_max = np.max([X.max(0), X_rotated.max(0)], 0)
    x1_grid, x2_grid = np.meshgrid(np.linspace(x1_min-0.1, x1_max+0.1, 100), np.linspace(x2_min-0.1, x2_max+0.1, 100))
    x_grid = np.stack([x1_grid.reshape(-1), x2_grid.reshape(-1)])
    return X, X_rotated, y, y_rotated, x_grid, x1_grid, x2_grid


def get_loader(source_X, target_X, source_y_task, target_y_task, batch_size=34):
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
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=True)
    
    return source_loader, target_loader, source_y_task, source_X, target_X, target_y_task


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc1(x))


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, fc_sizes=[10, 10]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -1


def fit(source_loader, target_loader, target_X, target_y_task,
        feature_extractor, domain_classifier, task_classifier, criterion,
        feature_optimizer, domain_optimizer, task_optimizer, num_epochs=1000, is_timeseries=False):
    
    reverse_grad = ReverseGradient.apply
    # TODO: Understand torch.autograd.Function.apply
    loss_domains = []
    loss_tasks = []
    loss_task_evals = []

    for _ in range(num_epochs):
        feature_extractor.train()
        task_classifier.train()

        for (source_X_batch, source_Y_batch), (target_X_batch, target_y_domain_batch) in zip(source_loader, target_loader):
            # 0. Data
            source_X_batch = source_X_batch
            source_y_task_batch = source_Y_batch[:, 0]
            source_y_domain_batch = source_Y_batch[:, 1]
            target_X_batch = target_X_batch
            target_y_domain_batch = target_y_domain_batch

            # 1. Forward
            # 1.1 Feature Extractor
            source_X_batch, target_X_batch = feature_extractor(source_X_batch), feature_extractor(target_X_batch)

            # 1.2. Task Classifier
            if is_timeseries:
                n_days = source_X_batch.shape[0]//16
                source_X_batch = source_X_batch.reshape(n_days, 16, source_X_batch.shape[1])
                pred_y_task = task_classifier(source_X_batch, DEVICE)
            else:
                pred_y_task = task_classifier(source_X_batch)

            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            loss_task = criterion(pred_y_task, source_y_task_batch)
            loss_tasks.append(loss_task.item())

            # 1.3. Domain Classifier
            source_X_batch, target_X_batch = reverse_grad(source_X_batch), reverse_grad(target_X_batch)
            pred_source_y_domain, pred_target_y_domain = domain_classifier(source_X_batch), domain_classifier(target_X_batch)
            pred_source_y_domain, pred_target_y_domain = torch.sigmoid(pred_source_y_domain).reshape(-1), torch.sigmoid(pred_target_y_domain).reshape(-1)

            loss_domain = criterion(pred_source_y_domain, source_y_domain_batch)
            loss_domain += criterion(pred_target_y_domain, target_y_domain_batch)
            loss_domains.append(loss_domain.item())

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

            if is_timeseries:
                n_days = target_feature_eval.shape[0]//16
                target_feature_eval = target_feature_eval.reshape(n_days, 16, target_feature_eval.shape[1])
                pred_y_task_eval = task_classifier(target_feature_eval, DEVICE)
            else:
                pred_y_task_eval = task_classifier(target_feature_eval)

            pred_y_task_eval = torch.sigmoid(pred_y_task_eval).reshape(-1)
            loss_task_eval =  criterion(pred_y_task_eval, target_y_task)
        loss_task_evals.append(loss_task_eval.item())
    
    # 4. Trace Each Loss
    plt.plot(loss_domains, label="loss_domain")
    plt.plot(loss_tasks, label="loss_task")
    plt.xlabel("batch")
    plt.ylabel("binary cross entropy loss")
    plt.legend()
    
    plt.figure()
    plt.plot(loss_task_evals, label="loss_task_eval")
    plt.xlabel("epoch")
    plt.ylabel("binary cross entropy loss")
    plt.legend()
    return feature_extractor, task_classifier


def fit_without_adaptation(source_loader, task_classifier, task_optimizer, criterion, num_epochs=1000):
    for _ in range(num_epochs):
        for source_X_batch, source_Y_batch in source_loader:
            # Prep Data
            source_X_batch = source_X_batch
            source_y_task_batch = source_Y_batch[:, 0]

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


def visualize_tSNE(target_feature_eval, source_X, feature_extractor):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    # TODO: Understand Argumetns for t-SNE

    target_feature_eval = target_feature_eval.cpu().detach().numpy()
    source_feature = feature_extractor(source_X)
    source_feature = source_feature.cpu().detach().numpy()

    target_feature_tsne = tsne.fit_transform(target_feature_eval)
    source_feature_tsne = tsne.fit_transform(source_feature)
    # TODO: Understand t-SNE Algo 

    plt.scatter(source_feature_tsne[:, 0], source_feature_tsne[:, 1], label="Source")
    plt.scatter(target_feature_tsne[:, 0], target_feature_tsne[:, 1], label="Target")
    plt.legend()


def raytune_trainer(config, options):
    # 1. Get Data from Options
    source_loader, target_loader, source_X, target_X, target_y_task = options.values()

    # 2. Instantiate Feature Extractor, Domain Classifier, Task Classifier
    num_domains = 1
    num_classes = 1
    feature_extractor = Encoder(input_size=source_X.shape[1], output_size=config["hidden_size"]).to(DEVICE)
    domain_classifier = Decoder(input_size=config["hidden_size"], output_size=num_domains,
                                fc_sizes=[config["domain_fc1_size"], config["domain_fc2_size"]]).to(DEVICE)
    task_classifier = Decoder(input_size=config["hidden_size"], output_size=num_classes,
                              fc_sizes=[config["task_fc1_size"], config["task_fc2_size"]]).to(DEVICE)

    # 3. Instantiate Criterion, Optimizer
    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=config["feature_learning_rate"],
                                   weight_decay=config["feature_weight_decay"], eps=config["feature_eps"])
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=config["domain_learning_rate"],
                                  weight_decay=config["domain_weight_decay"], eps=config["domain_eps"])
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=config["task_learning_rate"],
                                weight_decay=config["task_weight_decay"], eps=config["task_eps"])

    # 4. Domain Invariant Learning
    # TODO: Refactoring, Mostly Duplicated with "fit"
    reverse_grad = ReverseGradient.apply
    # TODO: Understand torch.autograd.Function.apply
    for _ in range(1000):
        feature_extractor.train()
        task_classifier.train()

        for (source_X_batch, source_Y_batch), (target_X_batch, target_y_domain_batch) in zip(source_loader, target_loader):
            # 4.0. Data
            source_X_batch = source_X_batch
            source_y_task_batch = source_Y_batch[:, 0]
            source_y_domain_batch = source_Y_batch[:, 1]
            target_X_batch = target_X_batch
            target_y_domain_batch = target_y_domain_batch

            # 4.1. Forward
            # 4.1.1 Feature Extractor
            source_X_batch, target_X_batch = feature_extractor(source_X_batch), feature_extractor(target_X_batch)

            # 4.1.2. Task Classifier
            pred_y_task = task_classifier(source_X_batch)
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            loss_task = criterion(pred_y_task, source_y_task_batch)

            # 4.1.3. Domain Classifier
            source_X_batch, target_X_batch = reverse_grad(source_X_batch), reverse_grad(target_X_batch)
            pred_source_y_domain, pred_target_y_domain = domain_classifier(source_X_batch), domain_classifier(target_X_batch)
            pred_source_y_domain, pred_target_y_domain = torch.sigmoid(pred_source_y_domain).reshape(-1), torch.sigmoid(pred_target_y_domain).reshape(-1)

            loss_domain = criterion(pred_source_y_domain, source_y_domain_batch)
            loss_domain += criterion(pred_target_y_domain, target_y_domain_batch)

            # 4.2. Backward, Update Params
            domain_optimizer.zero_grad()
            task_optimizer.zero_grad()
            feature_optimizer.zero_grad()

            loss_domain.backward(retain_graph = True)
            loss_task.backward() 

            domain_optimizer.step()
            task_optimizer.step()
            feature_optimizer.step()

        # 4.3. Evaluation
        feature_extractor.eval()
        task_classifier.eval()

        with torch.no_grad():
            target_feature_eval = feature_extractor(target_X)
            pred_y_task_eval = task_classifier(target_feature_eval)
            pred_y_task_eval = torch.sigmoid(pred_y_task_eval).reshape(-1)
            loss_task_eval =  criterion(pred_y_task_eval, target_y_task)
        tune.report(loss=loss_task_eval.item())