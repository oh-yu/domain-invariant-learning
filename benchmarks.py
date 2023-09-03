import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim

import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDXS = [1, 2, 3]


def main(source_idx, target_idx, winter_idx, summer_idx):
    # Cross Season Cross Household DA Once
    ## Prepare Data
    train_source_X = pd.read_csv(f"./deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./deep_occupancy_detection/data/{target_idx}_X_train.csv")

    train_source_y_task = pd.read_csv(f"./deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season==winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)

    train_source_X = train_source_X[train_source_X.Season==winter_idx]
    target_X = target_X[target_X.Season==summer_idx]

    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    target_X = scaler.fit_transform(target_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)
    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
    source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)
    # TODO: Update utils.get_loader's docstring

    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(device)
    test_target_y_task = test_target_y_task.to(device)


    ## Instantiate Feature Extractor, Domain Classifier, Task Classifier
    hidden_size = 128
    num_domains = 1
    num_classes = 1

    feature_extractor = utils.Conv1d(input_size=train_source_X.shape[2]).to(device)
    domain_classifier = utils.Decoder(input_size=hidden_size, output_size=num_domains).to(device)
    task_classifier = utils.Decoder(input_size=hidden_size, output_size=num_classes).to(device)

    learning_rate = 0.0001

    criterion = nn.BCELoss()
    feature_optimizer = optim.Adam(feature_extractor.parameters(), lr=learning_rate)
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=learning_rate)
    task_optimizer = optim.Adam(task_classifier.parameters(), lr=learning_rate)

    num_epochs = 300
    feature_extractor, task_classifier, _ = utils.fit(source_loader, target_loader, test_target_X, test_target_y_task,
                                                      feature_extractor, domain_classifier, task_classifier, criterion,
                                                      feature_optimizer, domain_optimizer, task_optimizer, num_epochs=num_epochs, is_timeseries=False)
    
    pred_y_task = task_classifier(feature_extractor(test_target_X))
    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
    pred_y_task = pred_y_task > 0.5
    acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
    return acc


if __name__ == "__main__":
    accs = []
    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            running_acc = 0
            if i == j:
                continue
            for _ in range(10):
                acc = main(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
                running_acc += acc.item()
            print(f"({i}, w) -> ({j}, s): {running_acc/10}")
            accs.append(running_acc/10)
    
    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            running_acc = 0
            if i == j:
                continue
            for _ in range(10):
                acc = main(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
                running_acc += acc.item()
            print(f"({i}, s) -> ({j}, w): {running_acc/10}")
            accs.append(running_acc/10)
    
    print(f"Average: {sum(accs)/len(accs)}")