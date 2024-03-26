import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold()
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from ..utils import utils
from ..models import IsihDanns, Codats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDX = [1, 2, 3]
SEASON_IDX = [0, 1]

class CoDATS_F_C(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.conv1d = utils.Conv1d(input_size=input_size).to(DEVICE)
        self.decoder = utils.Decoder(input_size=128, output_size=1).to(DEVICE)
    def forward(self, x):
        return self.decoder(self.conv1d(x))

def isih_da(source_idx=2, season_idx=0, n_splits:int=5):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_X = train_source_X.copy()
    tmp_list = [i for i in range(16, 44, 1)]
    tmp_list += [12, 13, 14, 15]
    tmp_list = tmp_list * int(train_source_X.shape[0]/32)
    target_X["Time"] = tmp_list
    target_y_task = train_source_y_task

    target_prime_X = train_source_X.copy()
    tmp_list = [i for i in range(18, 44, 1)]
    tmp_list += [12, 13, 14, 15, 16, 17]
    tmp_list = tmp_list * int(train_source_X.shape[0]/32)
    target_prime_X["Time"] = tmp_list
    target_prime_y_task = train_source_y_task

    train_source_y_task = train_source_y_task.values.reshape(-1)
    target_y_task = target_y_task.values.reshape(-1)
    target_prime_y_task = target_prime_y_task.values.reshape(-1)

    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    target_X = scaler.fit_transform(target_X)

    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X, target_X, target_y_task, target_y_task
    source_loader, target_loader, train_source_y_task, train_source_X, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)

    isih_dann = IsihDanns(input_size=train_source_X.shape[2], hidden_size=128, lr_dim1=0.0001, lr_dim2=0.00005, 
                          num_epochs_dim1=200, num_epochs_dim2=100)
    isih_dann.fit_1st_dim(source_loader, target_loader, test_target_X, test_target_y_task)
    pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=True)
    pred_y_task = pred_y_task > 0.5
    train_source_X = target_X
    train_source_y_task = pred_y_task.cpu().detach().numpy()
    target_X = target_prime_X
    target_y_task = target_prime_y_task

    target_X = scaler.fit_transform(target_X)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    kfold = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in kfold.split(target_X):
        train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X[train_idx], target_X[test_idx], target_y_task[train_idx], target_y_task[test_idx]
        source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)
        ## isih-DA fit, predict for 2nd dimension
        isih_dann.fit_2nd_dim(source_loader, target_loader, test_target_X, test_target_y_task)
        pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=False)

        # Algo3. Evaluation
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs)/n_splits
    
def codats(source_idx=2, season_idx=0, n_splits:int=5):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_prime_X = train_source_X.copy()
    tmp_list = [i for i in range(18, 44, 1)]
    tmp_list += [12, 13, 14, 15, 16, 17]
    tmp_list = tmp_list * int(train_source_X.shape[0]/32)
    target_prime_X["Time"] = tmp_list
    target_prime_y_task = train_source_y_task

    train_source_y_task = train_source_y_task.values.reshape(-1)
    target_prime_y_task = target_prime_y_task.values.reshape(-1)
    
    target_X, target_y_task = target_prime_X, target_prime_y_task
    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    target_X = scaler.fit_transform(target_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    kfold = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in kfold.split(target_X):
        train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X[train_idx], target_X[test_idx], target_y_task[train_idx], target_y_task[test_idx]
        source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)
        # TODO: Update utils.get_loader's docstring

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)


        ## CoDATS fit, predict
        codats = Codats(input_size=train_source_X.shape[2], hidden_size=128, lr=0.0001, num_epochs=300)
        codats.fit(source_loader, target_loader, test_target_X, test_target_y_task)
        pred_y_task = codats.predict(test_target_X)

        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs)/n_splits


def without_adapt(source_idx=2, season_idx=0, n_splits:int=5):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_prime_X = train_source_X.copy()
    tmp_list = [i for i in range(18, 44, 1)]
    tmp_list += [12, 13, 14, 15, 16, 17]
    tmp_list = tmp_list * int(train_source_X.shape[0]/32)
    target_prime_X["Time"] = tmp_list
    target_prime_y_task = train_source_y_task

    train_source_y_task = train_source_y_task.values.reshape(-1)
    target_prime_y_task = target_prime_y_task.values.reshape(-1)
    
    target_X, target_y_task = target_prime_X, target_prime_y_task
    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    target_X = scaler.fit_transform(target_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    kfold = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in kfold.split(target_X):
        train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X[train_idx], target_X[test_idx], target_y_task[train_idx], target_y_task[test_idx]
        source_loader, _, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)
        # TODO: Update utils.get_loader's docstring

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)


        ## Without Adapt fit, predict
        without_adapt = CoDATS_F_C(input_size=train_source_X.shape[2])
        without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        without_adapt = utils.fit_without_adaptation(source_loader=source_loader, task_classifier=without_adapt,
                                                    task_optimizer=without_adapt_optimizer, criterion=criterion, num_epochs=300)
        pred_y = without_adapt(test_target_X)
        pred_y = torch.sigmoid(pred_y).reshape(-1)
        pred_y = pred_y > 0.5
        acc = sum(pred_y == test_target_y_task) / pred_y.shape[0]
        accs.append(acc.item())
    return sum(accs)/n_splits


def train_on_target(source_idx=2, season_idx=0, n_splits:int=5):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_prime_X = train_source_X.copy()
    tmp_list = [i for i in range(18, 44, 1)]
    tmp_list += [12, 13, 14, 15, 16, 17]
    tmp_list = tmp_list * int(train_source_X.shape[0]/32)
    target_prime_X["Time"] = tmp_list
    target_prime_y_task = train_source_y_task
    target_prime_y_task = target_prime_y_task.values.reshape(-1)
    
    target_X, target_y_task = target_prime_X, target_prime_y_task
    scaler = preprocessing.StandardScaler()
    target_X = scaler.fit_transform(target_X)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    kfold = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in kfold.split(target_X):
        train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X[train_idx], target_X[test_idx], target_y_task[train_idx], target_y_task[test_idx]

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        train_target_X = torch.tensor(train_target_X, dtype=torch.float32)
        train_target_y_task = torch.tensor(train_target_y_task, dtype=torch.float32)
        train_target_X = train_target_X.to(DEVICE)
        train_target_y_task = train_target_y_task.to(DEVICE)
        target_ds = TensorDataset(train_target_X, train_target_y_task)
        target_loader = DataLoader(target_ds, batch_size=32, shuffle=True)
        ## Train on Target fit, predict
        train_on_target = CoDATS_F_C(input_size=train_target_X.shape[2])
        train_on_target_optimizer = optim.Adam(train_on_target.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        for _ in range(300):
            for target_X_batch, target_y_task_batch in target_loader:
                # Forward
                pred_y_task = train_on_target(target_X_batch)
                pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
                loss_task = criterion(pred_y_task, target_y_task_batch)

                # Backward
                train_on_target_optimizer.zero_grad()
                loss_task.backward()
                # Update Params
                train_on_target_optimizer.step()
        pred_y_task = train_on_target(test_target_X)
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / pred_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs)/n_splits
    

def main():
    accs_isih_da = []
    accs_codats = []
    accs_without_adapt = []
    accs_train_on_target = []
    patterns = []
    for i in HOUSEHOLD_IDX:
        for j in SEASON_IDX:
            acc_isih_da = isih_da(source_idx=i, season_idx=j)
            acc_codats = codats(source_idx=i, season_idx=j)
            acc_without_adapt = without_adapt(source_idx=i, season_idx=j)
            acc_train_on_target = train_on_target(source_idx=i, season_idx=j)
            accs_isih_da.append(acc_isih_da)
            accs_codats.append(acc_codats)
            accs_without_adapt.append(acc_without_adapt)
            accs_train_on_target.append(acc_train_on_target)
            patterns.append(f"Household ID:{i}, Season:{j}")
    df = pd.DataFrame()
    df["patterns"] = patterns
    df["accs_isih_da"] = accs_isih_da
    df["accs_codats"] = accs_codats
    df["accs_without_adapt"] = accs_without_adapt
    df["accs_train_on_target"] = accs_train_on_target
    df.to_csv("ecodataset_synthetic_experiment.csv", index=False)

if __name__ == "__main__":
    main()