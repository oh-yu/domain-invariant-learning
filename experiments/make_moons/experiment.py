from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from absl import app, flags
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from ...algo import coral_algo, dann2D_algo, dann_algo, coral2D_algo, supervised_algo
from ...networks import Encoder, ThreeLayersDecoder, Danns2D
from ...utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
flags.DEFINE_integer("rotation_degree", -25, "rotation degree for target data")
flags.DEFINE_string("algo_name", "DANN", "Which algo to use, DANN or CoRAL")
flags.DEFINE_boolean(
    "is_RV_tuning",
    True,
    "Whether or not use Reverse Validation based free params tuning method(5.1.2 algo from DANN paper)",
)
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
    # hidden_size = 10
    # num_domains = 1
    # num_classes = 1

    # feature_extractor = Encoder(input_size=source_X.shape[1], output_size=hidden_size).to(device)
    # domain_classifier = ThreeLayersDecoder(
    #     input_size=hidden_size, output_size=num_domains, dropout_ratio=0, fc1_size=50, fc2_size=10
    # ).to(device)
    # task_classifier = ThreeLayersDecoder(
    #     input_size=hidden_size, output_size=num_classes, dropout_ratio=0, fc1_size=50, fc2_size=10
    # ).to(device)
    # learning_rate = 0.005

    # criterion = nn.BCELoss()
    # feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    # domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=learning_rate)
    # task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)

    # data = {
    #     "source_loader": source_loader,
    #     "target_loader": target_prime_loader,
    #     "target_X": target_prime_X,
    #     "target_y_task": target_prime_y_task,
    # }
    # if FLAGS.algo_name == "DANN":
    #     network = {
    #         "feature_extractor": feature_extractor,
    #         "domain_classifier": domain_classifier,
    #         "task_classifier": task_classifier,
    #         "criterion": criterion,
    #         "feature_optimizer": feature_optimizer,
    #         "domain_optimizer": domain_optimizer,
    #         "task_optimizer": task_optimizer,
    #     }
    #     config = {
    #         "num_epochs": 1000,
    #         "do_plot": True,
    #         "is_target_weights": True,
    #     }
    # elif FLAGS.algo_name == "CoRAL":
    #     network = {
    #         "feature_extractor": feature_extractor,
    #         "task_classifier": task_classifier,
    #         "criterion": criterion,
    #         "task_optimizer": task_optimizer,
    #         "feature_optimizer": feature_optimizer,
    #     }
    #     config = {
    #         "num_epochs": 1000,
    #         "alpha": 1,
    #         "do_plot": True,
    #     }
    # feature_extractor, task_classifier, _ = ALGORYTHMS[FLAGS.algo_name].fit(data, network, **config)

    # target_prime_feature_eval = feature_extractor(target_prime_X)
    # pred_y_task = task_classifier(target_prime_feature_eval)
    # pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    # pred_y_task = pred_y_task > 0.5

    # dann_acc = sum(pred_y_task == target_prime_y_task) / target_prime_y_task.shape[0]
    # print(f"DANNs Accuracy:{dann_acc}")

    # source_X = source_X.cpu()
    # target_prime_X = target_prime_X.cpu()

    # x_grid = torch.tensor(x_grid, dtype=torch.float32)
    # x_grid = x_grid.to(device)

    # x_grid_feature = feature_extractor(x_grid.T)
    # y_grid = task_classifier(x_grid_feature)
    # y_grid = torch.sigmoid(y_grid)
    # y_grid = y_grid.cpu().detach().numpy()

    # plt.figure()
    # plt.title("Domain Adaptation Boundary")
    # plt.xlabel("X1")
    # plt.ylabel("X2")
    # plt.scatter(source_X[:, 0], source_X[:, 1], c=source_y_task)
    # plt.scatter(target_prime_X[:, 0], target_prime_X[:, 1], c="black")
    # plt.contourf(x1_grid, x2_grid, y_grid.reshape(100, 100), alpha=0.3)
    # plt.colorbar()
    # plt.show()

    # 2D-DANNs
    danns_2d = Danns2D(experiment="make_moons")
    rv_scores = danns_2d.fit(source_loader, target_loader, target_prime_loader, target_prime_X, target_prime_y_task)
    print(rv_scores)


if __name__ == "__main__":
    app.run(main)
