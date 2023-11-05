import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim

from ..utils import utils
from ..models import IsihDanns, Codats
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDXS = [1, 2, 3]


def isih_da_house(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int) -> torch.Tensor:
    """
    Execute isih-DA (Household => Season) experiment.
    TODO: Attach ICASSP2024 REF
    """
    # Algo1. Inter-Households DA
    ## Prepare Data
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season==winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==winter_idx].values.reshape(-1)
    train_source_X = train_source_X[train_source_X.Season==winter_idx]
    target_X = target_X[target_X.Season==winter_idx]

    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    target_X = scaler.fit_transform(target_X)

    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X, target_X, target_y_task, target_y_task
    source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)

    ## isih-DA fit, predict for 1st dimension
    isih_dann = IsihDanns(input_size=train_source_X.shape[2], hidden_size=128, lr_dim1=0.0001, lr_dim2=0.00005, 
                          num_epochs_dim1=200, num_epochs_dim2=100)
    isih_dann.fit_1st_dim(source_loader, target_loader, test_target_X, test_target_y_task)
    pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=True)

    # Algo2. Inter-Seasons DA
    ## Prepare Data
    train_source_X = target_X
    train_source_y_task = pred_y_task.cpu().detach().numpy()
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)
    target_X = target_X[target_X.Season==summer_idx].values

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


def isih_da_season(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int) -> torch.Tensor:
    """
    Execute isih-DA (Season => Household) experiment.
    TODO: Attach ICASSP2024 REF
    """
    # Algo1. Inter-Seasons DA
    ## Prepare Data
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season==winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)
    train_source_X = train_source_X[train_source_X.Season==winter_idx]
    target_X = target_X[target_X.Season==summer_idx]

    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    target_X = scaler.fit_transform(target_X)

    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X, target_X, target_y_task, target_y_task
    source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)

    ## isih-DA fit, predict for 1st dimension
    isih_dann = IsihDanns(input_size=train_source_X.shape[2], hidden_size=128, lr_dim1=0.0001, lr_dim2=0.00005, 
                          num_epochs_dim1=200, num_epochs_dim2=100)
    isih_dann.fit_1st_dim(source_loader, target_loader, test_target_X, test_target_y_task)
    pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=True)

    # Algo2. Inter-Households DA
    ## Prepare Data
    train_source_X = target_X
    train_source_y_task = pred_y_task.cpu().detach().numpy()
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)
    target_X = target_X[target_X.Season==summer_idx].values

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


def codats(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int) -> torch.Tensor:
    """
    Execute CoDATS experiment.
    TODO: Attach ICASSP2024 REF
    """
    # Direct Inter-Seasons and Inter-Households DA
    ## Prepare Data
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")

    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season==winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)

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
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)


    ## CoDATS fit, predict
    codats = Codats(input_size=train_source_X.shape[2], hidden_size=128, lr=0.0001, num_epochs=300)
    codats.fit(source_loader, target_loader, test_target_X, test_target_y_task)
    pred_y_task = codats.predict(test_target_X)

    pred_y_task = pred_y_task > 0.5
    acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
    return acc


def main():
    isih_da_house_accs = []
    isih_da_season_accs = []
    codats_accs = []

    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            isih_da_house_running_acc = 0
            isih_da_season_running_acc = 0
            codats_running_acc = 0
            


            if i == j:
                continue
            for _ in range(10):
                isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
                isih_da_house_running_acc += isih_da_house_acc.item()

                isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
                isih_da_season_running_acc += isih_da_season_acc.item()

                codats_acc = codats(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
                codats_running_acc += codats_acc.item()


            print(f"({i}, w) -> ({j}, s)")
            print(f"isih-DA (Household => Season)| {isih_da_house_running_acc/10}")
            print(f"isih-DA (Season => Household)| {isih_da_season_running_acc/10}")
            print(f"CoDATS| {codats_running_acc/10}")
 
            isih_da_house_accs.append(isih_da_house_running_acc/10)
            isih_da_season_accs.append(isih_da_season_running_acc/10)
            codats_accs.append(codats_running_acc/10)
    
    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            isih_da_house_running_acc = 0
            isih_da_season_running_acc = 0
            codats_running_acc = 0

            if i == j:
                continue
            for _ in range(10):
                isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
                isih_da_house_running_acc += isih_da_house_acc.item()

                isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
                isih_da_season_running_acc += isih_da_season_acc.item()

                codats_acc = codats(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
                codats_running_acc += codats_acc.item()

            print(f"({i}, s) -> ({j}, w)")
            print(f"isih-DA (Household => Season)| {isih_da_house_running_acc/10}")
            print(f"isih-DA (Season => Household)| {isih_da_season_running_acc/10}")
            print(f"CoDATS| {codats_running_acc/10}")

            isih_da_house_accs.append(isih_da_house_running_acc/10)
            isih_da_season_accs.append(isih_da_season_running_acc/10)
            codats_accs.append(codats_running_acc/10)
    
    print(f"isih-DA (Household => Season) Average: {sum(isih_da_house_accs)/len(isih_da_house_accs)}")
    print(f"isih-DA (Season => Household) Average: {sum(isih_da_season_accs)/len(isih_da_season_accs)}")
    print(f"CoDATS Average: {sum(codats_accs)/len(codats_accs)}")

if __name__ == "__main__":
    main()