import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from ..utils import utils
from ..models import IsihDanns, Codats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDX = [1, 2, 3]
SEASON_IDX = [0, 1]

class CoDATS_F_C(nn.Module):
    def __init__(self, input_size: int, out_channels1: int = 128, out_channels2: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels1)
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(out_channels2)
        self.fc1 = nn.Linear(out_channels2, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[1])
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = torch.mean(x, dim=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def isih_da(source_idx=2, season_idx=0):
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

    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
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
    return acc
    
def codats(source_idx=2, season_idx=0):
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
    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
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
    return acc


def without_adapt(source_idx=2, season_idx=0):
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
    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
    source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)
    # TODO: Update utils.get_loader's docstring

    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)


    ## Without Adapt fit, predict
    without_adapt = CoDATS_F_C(input_size=train_source_X.shape[2])
    without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    without_adapt = utils.fit_without_adaptation(source_loader=source_loader, task_classifier=without_adapt,
                                                 task_optimizer=without_adapt_optimizer, criterion=criterion)
    pred_y = without_adapt(test_target_X)
    pred_y = torch.sigmoid(pred_y).reshape(-1)
    acc = sum(pred_y == test_target_y_task) / pred_y.shape[0]
    return acc
    

def main():
    accs_isih_da = []
    accs_codats = []
    accs_without_adapt = []
    patterns = []
    for i in HOUSEHOLD_IDX:
        for j in SEASON_IDX:
            runnning_acc_isih_da = 0
            runnning_acc_codats = 0
            running_acc_without_adapt = 0
            for _ in range(10):
                runnning_acc_isih_da += isih_da(source_idx=i, season_idx=j).item()
                runnning_acc_codats += codats(source_idx=i, season_idx=j).item()
                running_acc_without_adapt += without_adapt(source_idx=i, season_idx=j).item()
            accs_isih_da.append(runnning_acc_isih_da/10)
            accs_codats.append(runnning_acc_codats/10)
            accs_without_adapt.append(running_acc_without_adapt/10)
            patterns.append(f"Household ID:{i}, Season:{j}")
    df = pd.DataFrame()
    df["patterns"] = patterns
    df["accs_isih_da"] = accs_isih_da
    df["accs_codats"] = accs_codats
    df["accs_without_adapt"] = accs_without_adapt
    df.to_csv("ecodataset_synthetic_experiment.csv", index=False)

if __name__ == "__main__":
    main()