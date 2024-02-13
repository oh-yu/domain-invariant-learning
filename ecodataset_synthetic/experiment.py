import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim

from ..utils import utils
from ..models import IsihDanns, Codats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDX = [0, 1, 2]
SEASON_IDX = [0, 1]

def isih_da(source_idx=2, season_idx=0):
    train_source_X = pd.read_csv(f"./deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season == season_idx]
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
                          num_epochs_dim1=900, num_epochs_dim2=100)
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
    train_source_X = pd.read_csv(f"./deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season == season_idx]
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
    codats = Codats(input_size=train_source_X.shape[2], hidden_size=128, lr=0.0001, num_epochs=1000)
    codats.fit(source_loader, target_loader, test_target_X, test_target_y_task)
    pred_y_task = codats.predict(test_target_X)

    pred_y_task = pred_y_task > 0.5
    acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
    return acc
    

def main():
    accs_isih_da = []
    accs_codats = []
    patterns = []
    for i in HOUSEHOLD_IDX:
        for j in SEASON_IDX:
            runnning_acc_isih_da = 0
            runnning_acc_codats = 0
            for _ in range(10):
                runnning_acc_isih_da += isih_da(source_idx=i, season_idx=j).item()
                runnning_acc_codats += codats(source_idx=i, season_idx=j).item()
            accs_isih_da.append(runnning_acc_isih_da/10)
            accs_codats.append(runnning_acc_codats/10)
            patterns.append(f"Household ID:{i}, Season:{j}")
    df = pd.DataFrame()
    df["patterns"] = patterns
    df["accs_isih_da"] = accs_isih_da
    df["accs_codats"] = accs_codats
    df.to_csv("ecodataset_synthetic_experiment.csv")

if __name__ == "__main__":
    main()