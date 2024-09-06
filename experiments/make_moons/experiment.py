from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from absl import app, flags
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ...algo import coral_algo, dann2D_algo, dann_algo, supervised_algo
from ...networks import Encoder, ThreeLayersDecoder
from ...utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
flags.DEFINE_integer("rotation_degree", -25, "rotation degree for target data")
flags.DEFINE_string("algo_name", "DANN", "Which algo to use, DANN or CoRAL")
flags.mark_flag_as_required("rotation_degree")
flags.mark_flag_as_required("algo_name")


ALGORYTHMS = {
    "DANN": dann_algo,
    "CoRAL": coral_algo,
}


def main(argv):
    # Prepare Data
    (
        source_X,
        source_y_task,
        target_X,
        target_y_task,
        target_prime_X,
        target_prime_y_task,
        x_grid,
        x1_grid,
        x2_grid,
    ) = utils.get_source_target_from_make_moons(rotation_degree=FLAGS.rotation_degree)
    source_loader, target_prime_loader, source_y_task, source_X, target_prime_X, target_prime_y_task = utils.get_loader(
        source_X, target_prime_X, source_y_task, target_prime_y_task
    )
    target_X = torch.tensor(target_X, dtype=torch.float32).to(utils.DEVICE)
    target_y_task = torch.tensor(target_y_task, dtype=torch.float32).to(utils.DEVICE)
    target_ds = TensorDataset(target_X, torch.ones_like(target_y_task))
    target_loader = DataLoader(target_ds, batch_size=34, shuffle=False)

    # DANNs
    hidden_size = 10
    num_domains = 1
    num_classes = 1

    feature_extractor = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(device)
    domain_classifier = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    task_classifier = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    learning_rate = 0.005

    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=learning_rate)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)

    data = {
        "source_loader": source_loader,
        "target_loader": target_prime_loader,
        "target_X": target_prime_X,
        "target_y_task": target_prime_y_task,
    }
    if FLAGS.algo_name == "DANN":
        network = {
            "feature_extractor": feature_extractor,
            "domain_classifier": domain_classifier,
            "task_classifier": task_classifier,
            "criterion": criterion,
            "feature_optimizer": feature_optimizer,
            "domain_optimizer": domain_optimizer,
            "task_optimizer": task_optimizer,
        }
        config = {
            "num_epochs": 1000,
            "do_plot": True,
            "is_target_weights": True,
        }
    elif FLAGS.algo_name == "CoRAL":
        network = {
            "feature_extractor": feature_extractor,
            "task_classifier": task_classifier,
            "criterion": criterion,
            "task_optimizer": task_optimizer,
            "feature_optimizer": feature_optimizer,
        }
        config = {
            "num_epochs": 1000,
            "alpha": 1,
            "is_changing_lr": True,
            "epoch_thr_for_changing_lr": 200,
            "changed_lrs": [0.00005, 0.00005],
            "do_plot": True,
        }
    feature_extractor, task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    target_prime_feature_eval = feature_extractor(target_prime_X)
    pred_y_task = task_classifier(target_prime_feature_eval)
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5

    dann_acc = sum(pred_y_task == target_prime_y_task) / target_prime_y_task.shape[0]
    print(f"DANNs Accuracy:{dann_acc}")

    source_X = source_X.cpu()
    target_prime_X = target_prime_X.cpu()

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
    plt.scatter(target_prime_X[:, 0], target_prime_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()

    # 2D-DANNs
    hidden_size = 10
    num_domains = 1
    num_classes = 1
    feature_extractor_dim12 = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(device)
    domain_classifier_dim1 = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    domain_classifier_dim2 = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    task_classifier = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor_dim12.parameters(), lr=learning_rate)
    domain_optimizer_dim1 = optim.Adam(domain_classifier_dim1.parameters(), lr=learning_rate)
    domain_optimizer_dim2 = optim.Adam(domain_classifier_dim2.parameters(), lr=learning_rate)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)

    data = {
        "source_loader": source_loader,
        "target_loader": target_loader,
        "target_prime_loader": target_prime_loader,
        "target_prime_X": target_prime_X.to(utils.DEVICE),
        "target_prime_y_task": target_prime_y_task,
    }
    network = {
        "feature_extractor": feature_extractor_dim12,
        "domain_classifier_dim1": domain_classifier_dim1,
        "domain_classifier_dim2": domain_classifier_dim2,
        "task_classifier": task_classifier,
        "criterion": criterion,
        "feature_optimizer": feature_optimizer,
        "domain_optimizer_dim1": domain_optimizer_dim1,
        "domain_optimizer_dim2": domain_optimizer_dim2,
        "task_optimizer": task_optimizer,
    }
    config = {"num_epochs": 1000, "do_plot": True}
    feature_extractor_dim12, task_classifier, _ = dann2D_algo.fit(data, network, **config)
    y_grid = task_classifier.predict_proba(feature_extractor_dim12(x_grid.T)).cpu().detach().numpy()
    pred_y_task = task_classifier.predict(feature_extractor_dim12(target_prime_X.to(device)))
    danns_2D_acc = sum(pred_y_task == target_prime_y_task) / len(pred_y_task)
    print(f"2D-DANNs Accuracy: {danns_2D_acc.item()}")
    plt.figure()
    plt.title("2D-DANN Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_prime_X[:, 0], target_prime_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()

    # step-by-step DANNs
    hidden_size = 10
    num_domains = 1
    num_classes = 1
    feature_extractor_dim12 = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(device)
    domain_classifier_dim1 = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    task_classifier_dim1 = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    domain_classifier_dim2 = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)
    task_classifier_dim2 = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)

    criterion = nn.BCELoss()
    feature_optimizer_dim1 = optim.Adam(feature_extractor_dim12.parameters(), lr=learning_rate)
    domain_optimizer_dim1 = optim.Adam(domain_classifier_dim1.parameters(), lr=learning_rate)
    task_optimizer_dim1 = optim.Adam(task_classifier_dim1.parameters(), lr=learning_rate)
    feature_optimizer_dim2 = optim.Adam(feature_extractor_dim12.parameters(), lr=learning_rate)
    domain_optimizer_dim2 = optim.Adam(domain_classifier_dim2.parameters(), lr=learning_rate)
    task_optimizer_dim2 = optim.Adam(task_classifier_dim2.parameters(), lr=learning_rate)
    ## 1st dim

    data = {
        "source_loader": source_loader,
        "target_loader": target_loader,
        "target_X": target_X,
        "target_y_task": target_y_task,
    }

    if FLAGS.algo_name == "DANN":
        network = {
            "feature_extractor": feature_extractor_dim12,
            "domain_classifier": domain_classifier_dim1,
            "task_classifier": task_classifier_dim1,
            "criterion": criterion,
            "feature_optimizer": feature_optimizer_dim1,
            "domain_optimizer": domain_optimizer_dim1,
            "task_optimizer": task_optimizer_dim1,
        }
        config = {
            "num_epochs": 200,
            "do_plot": True,
            "is_target_weights": True,
        }

    elif FLAGS.algo_name == "CoRAL":
        network = {
            "feature_extractor": feature_extractor_dim12,
            "task_classifier": task_classifier_dim1,
            "criterion": criterion,
            "task_optimizer": task_optimizer_dim1,
            "feature_optimizer": feature_optimizer_dim1,
        }
        config = {"num_epochs": 200, "alpha": 1, "do_plot": True}
    feature_extractor_dim12, task_classifier_dim1, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    target_feature_eval = feature_extractor_dim12(target_X)
    pred_y_task = task_classifier_dim1(target_feature_eval)
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)

    ## 2nd dim
    target_ds = TensorDataset(
        target_X,
        torch.cat([pred_y_task.detach().reshape(-1, 1), torch.zeros_like(target_y_task).reshape(-1, 1)], dim=1),
    )
    target_loader = DataLoader(target_ds, batch_size=34, shuffle=False)

    data = {
        "source_loader": target_loader,
        "target_loader": target_prime_loader,
        "target_X": target_prime_X.to(utils.DEVICE),
        "target_y_task": target_prime_y_task,
    }
    if FLAGS.algo_name == "DANN":
        network = {
            "feature_extractor": feature_extractor_dim12,
            "domain_classifier": domain_classifier_dim2,
            "task_classifier": task_classifier_dim2,
            "criterion": criterion,
            "feature_optimizer": feature_optimizer_dim2,
            "domain_optimizer": domain_optimizer_dim2,
            "task_optimizer": task_optimizer_dim2,
        }
        config = {"num_epochs": 800, "do_plot": True, "is_target_weights": True, "is_psuedo_weights": True}

    elif FLAGS.algo_name == "CoRAL":
        network = {
            "feature_extractor": feature_extractor_dim12,
            "task_classifier": task_classifier_dim2,
            "criterion": criterion,
            "task_optimizer": task_optimizer_dim2,
            "feature_optimizer": feature_optimizer_dim2,
        }
        config = {"num_epochs": 800, "alpha": 1, "is_psuedo_weights": True, "do_plot": True}

    feature_extractor_dim12, task_classifier_dim2, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    ## Eval
    pred_y_task = task_classifier_dim2(feature_extractor_dim12(target_prime_X.to(utils.DEVICE)))
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5
    stepbystep_dann_acc = sum(pred_y_task == target_prime_y_task) / target_prime_y_task.shape[0]
    print(f"step-by-step DANNs Accuracy:{stepbystep_dann_acc}")

    x_grid_feature = feature_extractor_dim12(x_grid.T)
    y_grid = task_classifier_dim2(x_grid_feature)
    y_grid = torch.sigmoid(y_grid)
    y_grid = y_grid.cpu().detach().numpy()

    plt.figure()
    plt.title("step-by-step DANNs Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_prime_X[:, 0], target_prime_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()

    # Without Adaptation
    feature_extractor_withoutadapt = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(device)
    task_classifier = ThreeLayersDecoder(
        input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10
    ).to(device)

    optimizer = optim.Adam(list(task_classifier.parameters())+list(feature_extractor_withoutadapt.parameters()), lr=learning_rate)
    data = {
        "loader": source_loader
    }
    network = {
        "decoder": task_classifier,
        "encoder": feature_extractor_withoutadapt,
        "optimizer": optimizer,
        "criterion": criterion
    }
    config = {
        "use_source_loader": True,
        "num_epochs": 1000
    }
    task_classifier, feature_extractor_withoutadapt = supervised_algo.fit(data, network, **config)
    pred_y_task = task_classifier(feature_extractor_withoutadapt(target_prime_X.to(device)))
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5
    without_adapt_acc = sum(pred_y_task == target_prime_y_task) / target_prime_y_task.shape[0]
    print(f"Without Adaptation Accuracy:{without_adapt_acc}")

    y_grid = task_classifier(feature_extractor_withoutadapt(x_grid.T))
    y_grid = torch.sigmoid(y_grid)
    y_grid = y_grid.cpu().detach().numpy()

    plt.figure()
    plt.title("Without Adaptation Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    plt.scatter(target_prime_X[:, 0], target_prime_X[:, 1], c="black")
    plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    plt.colorbar()
    plt.show()

    # t-SNE Visualization for Extracted Feature
    target_prime_feature_eval = target_prime_feature_eval.cpu().detach().numpy()
    source_feature = feature_extractor(source_X.to(device))
    source_feature = source_feature.cpu().detach().numpy()

    utils.visualize_tSNE(target_prime_feature_eval, source_feature)

    # to csv
    df = pd.DataFrame()
    df["PAT"] = [f"source->{FLAGS.rotation_degree}rotated->{FLAGS.rotation_degree*2}rotated"]
    df["2D-DANNs"] = [danns_2D_acc.item()]
    df["stepbystep-DANNs"] = [stepbystep_dann_acc.item()]
    df["DANNs"] = [dann_acc.item()]
    df["WithoutAdapt"] = [without_adapt_acc.item()]
    df.to_csv(f"make_moons_{str(datetime.now())}_{FLAGS.algo_name}.csv", index=False)


if __name__ == "__main__":
    app.run(main)
