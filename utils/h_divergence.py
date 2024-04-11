import pandas as pd
import torch
from sklearn import preprocessing
from torch import optim

from ..networks import Conv1d, Decoder
from .utils import *

DEVICE = DEVICE


def h_divergence(source_loader, target_loader, source_X, target_X):
    # 1. Initialize H
    feature_extractor = Conv1d(input_size=train_source_X.shape[2]).to(DEVICE)
    domain_classifier = Decoder(input_size=128, output_size=1).to(DEVICE)
    criterion = torch.nn.BCELoss()

    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=0.0001)
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=0.0001)

    # 2. Minimize domain classification error
    for epoch in range(30):
        for i, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
            source_X_tmp, source_y = source_data
            source_y = source_y[:, 1]
            target_X, target_y = target_data

            source_feature = feature_extractor(source_X_tmp)
            target_feature = feature_extractor(target_X)

            source_output = domain_classifier(source_feature)
            target_output = domain_classifier(target_feature)

            source_output = torch.sigmoid(source_output).reshape(-1)
            target_output = torch.sigmoid(target_output).reshape(-1)
            loss = criterion(source_output, torch.zeros_like(source_output))
            loss += criterion(target_output, torch.ones_like(target_output))
            feature_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            loss.backward()
            feature_optimizer.step()
            domain_optimizer.step()
        if epoch % 10 == 0:
            print(f"Loss: {loss.item()}")

    # 3. Compute H-divergence
    pred_y = domain_classifier(feature_extractor(source_X))
    pred_y = torch.sigmoid(pred_y).reshape(-1)
    pred_y = pred_y > 0.5
    source_y = torch.zeros_like(pred_y)
    acc_source = sum(pred_y == source_y) / pred_y.shape[0]
    err_source = 1 - acc_source

    pred_y = domain_classifier(feature_extractor(target_X))
    pred_y = torch.sigmoid(pred_y).reshape(-1)
    pred_y = pred_y > 0.5
    target_y = torch.ones_like(pred_y)
    acc_target = sum(pred_y == target_y) / pred_y.shape[0]
    err_target = 1 - acc_target
    return 2 * (1 - (err_source + err_target))


if __name__ == "__main__":
    source_idx = 1
    target_idx = 2
    winter_idx = 0

    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[
        target_X.Season == winter_idx
    ].values.reshape(-1)
    train_source_X = train_source_X[train_source_X.Season == winter_idx]
    target_X = target_X[target_X.Season == winter_idx]

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    target_X = scaler.transform(target_X)

    train_source_X, train_source_y_task = apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = apply_sliding_window(target_X, target_y_task, filter_len=6)

    source_loader, target_loader, _, source_X, target_X, _ = get_loader(
        train_source_X, target_X, train_source_y_task, target_y_task, shuffle=True
    )
    print(h_divergence(source_loader, target_loader, source_X, target_X))
