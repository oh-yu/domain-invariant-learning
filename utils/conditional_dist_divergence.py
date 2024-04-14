import pandas as pd
import torch
from sklearn import preprocessing
from torch import optim

from ..networks import Conv1d, Decoder
from .utils import DEVICE, apply_sliding_window, get_loader


def conditional_dist_divergence(source_loader, target_X, target_y_task):
    """
    TODO: Attach Paper
    """
    # 1. Init Model
    feature_extractor = Conv1d(input_size=target_X.shape[2]).to(DEVICE)
    task_classifier = Decoder(input_size=128, output_size=1).to(DEVICE)
    criterion = torch.nn.BCELoss()

    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=0.0001)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=0.0001)

    # 2. Train on Source Data
    for epoch in range(100):
        for source_data in source_loader:
            source_X, source_y = source_data
            source_y_task = source_y[:, 0]

            source_feature = feature_extractor(source_X)
            source_output = task_classifier(source_feature)
            source_output = torch.sigmoid(source_output).reshape(-1)

            loss = criterion(source_output, source_y_task)

            feature_optimizer.zero_grad()
            task_optimizer.zero_grad()
            loss.backward()
            feature_optimizer.step()
            task_optimizer.step()
        if epoch % 50 == 0:
            print(f"Loss: {loss.item()}")

    # 3. test on target data
    pred_y = task_classifier(feature_extractor(target_X))
    pred_y = torch.sigmoid(pred_y).reshape(-1)
    pred_y = pred_y > 0.5
    acc = sum(pred_y == target_y_task) / pred_y.shape[0]
    return acc


if __name__ == "__main__":
    source_idx = 3
    target_idx = 1
    winter_idx = 0
    summer_idx = 0

    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[
        target_X.Season == summer_idx
    ].values.reshape(-1)
    train_source_X = train_source_X[train_source_X.Season == winter_idx]
    target_X = target_X[target_X.Season == summer_idx]

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    target_X = scaler.transform(target_X)

    train_source_X, train_source_y_task = apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = apply_sliding_window(target_X, target_y_task, filter_len=6)
    source_loader, _, _, _, target_X, target_y_task = get_loader(
        train_source_X, target_X, train_source_y_task, target_y_task, shuffle=True
    )

    test_thr = int(target_X.shape[0] / 2)
    test_target_X = target_X[test_thr:]
    test_target_y_task = target_y_task[test_thr:]

    print(conditional_dist_divergence(source_loader, target_X, target_y_task))
