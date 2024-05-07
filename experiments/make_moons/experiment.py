import matplotlib.pyplot as plt
import torch
from absl import app, flags
from torch import nn, optim

from ...algo import algo
from ...networks import TaskDecoder, DomainDecoder, Encoder
from ...utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
flags.DEFINE_integer("rotation_degree", -30, "rotation degree for target data")
flags.mark_flag_as_required("rotation_degree")


def main(argv):
    # Prepare Data
    (
        source_X,
        target_X,
        source_y_task,
        target_y_task,
        x_grid,
        x1_grid,
        x2_grid,
    ) = utils.get_source_target_from_make_moons(rotation_degree=FLAGS.rotation_degree)
    source_loader, target_loader, source_y_task, source_X, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task
    )

    # Instantiate Feature Extractor, Domain Classifier, Task Classifier
    hidden_size = 10
    num_domains = 1
    num_classes = 1

    feature_extractor = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(device)
    domain_classifier = DomainDecoder(input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10).to(device)
    task_classifier = DomainDecoder(input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10).to(device)
    learning_rate = 0.001

    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=learning_rate)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)

    # Domain Invariant Learning
    num_epochs = 1000
    feature_extractor, task_classifier, accs = algo.fit(
        source_loader,
        target_loader,
        target_X,
        target_y_task,
        feature_extractor,
        domain_classifier,
        task_classifier,
        criterion,
        feature_optimizer,
        domain_optimizer,
        task_optimizer,
        num_epochs=num_epochs,
        do_plot=True,
    )
    target_feature_eval = feature_extractor(target_X)
    pred_y_task = task_classifier(target_feature_eval)
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5

    acc = sum(pred_y_task == target_y_task) / target_y_task.shape[0]
    print(f"Accuracy:{acc}")

    source_X = source_X.cpu()
    target_X = target_X.cpu()

    x_grid = torch.tensor(x_grid, dtype=torch.float32)
    x_grid = x_grid.to(device)

    x_grid_feature = feature_extractor(x_grid.T)
    y_grid = task_classifier(x_grid_feature)
    y_grid = torch.sigmoid(y_grid)
    y_grid = y_grid.cpu().detach().numpy()

    plt.figure()
    plt.title("Domain Adaptation Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_X[:, 0], target_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()

    # Without Adaptation
    task_classifier = DomainDecoder(input_size=source_X.shape[1], output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10).to(device)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)
    task_classifier = utils.fit_without_adaptation(
        source_loader, task_classifier, task_optimizer, criterion, num_epochs
    )
    pred_y_task = task_classifier(target_X.to(device))
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5
    acc = sum(pred_y_task == target_y_task) / target_y_task.shape[0]
    print(f"Without Adaptation Accuracy:{acc}")

    y_grid = task_classifier(x_grid.T)
    y_grid = torch.sigmoid(y_grid)
    y_grid = y_grid.cpu().detach().numpy()

    plt.figure()
    plt.title("Without Adaptation Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_X[:, 0], target_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()

    # t-SNE Visualization for Extracted Feature
    target_feature_eval = target_feature_eval.cpu().detach().numpy()
    source_feature = feature_extractor(source_X.to(device))
    source_feature = source_feature.cpu().detach().numpy()

    utils.visualize_tSNE(target_feature_eval, source_feature)


if __name__ == "__main__":
    app.run(main)
