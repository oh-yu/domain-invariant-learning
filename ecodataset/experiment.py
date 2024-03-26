import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import torch
from torch import nn
from torch import optim

from ..utils import utils
from ..models import IsihDanns, Codats, CoDATS_F_C
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDXS = [1, 2, 3]


def isih_da_house(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5) -> torch.Tensor:
    """
    Execute isih-DA (Household => Season) experiment.
    TODO: Attach paper
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

    kf = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in kf.split(target_X):
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


def isih_da_season(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5) -> torch.Tensor:
    """
    Execute isih-DA (Season => Household) experiment.
    TODO: Attach paper
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


def codats(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5) -> torch.Tensor:
    """
    Execute CoDATS experiment.
    TODO: Attach paper
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


def without_adapt(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5) -> float:
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

    kfold = KFold(n_splits=n_splits, shuffle=False)
    accs = []
    for train_idx, test_idx in kfold.split(target_X):
        train_target_X, test_target_X, train_target_y_task, test_target_y_task = target_X[train_idx], target_X[test_idx], target_y_task[train_idx], target_y_task[test_idx]
        source_loader, _, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        without_adapt = CoDATS_F_C(input_size=train_source_X.shape[2])
        without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        without_adapt = utils.fit_without_adaptation(source_loader=source_loader, task_classifier=without_adapt, task_optimizer=without_adapt_optimizer, criterion=criterion, num_epochs=300)
        pred_y = without_adapt(test_target_X)
        pred_y = torch.sigmoid(pred_y).reshape(-1)
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs)/n_splits

def main():
    isih_da_house_accs = []
    isih_da_season_accs = []
    codats_accs = []
    without_adapt_accs = []
    df = pd.DataFrame()
    patterns = []

    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:

            if i == j:
                continue
            isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            codats_acc = codats(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            without_adapt_acc = without_adapt(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
 
            isih_da_house_accs.append(isih_da_house_acc)
            isih_da_season_accs.append(isih_da_season_acc)
            codats_accs.append(codats_acc)
            without_adapt_accs.append(without_adapt_acc)
            patterns.append(f"({i}, w) -> ({j}, s)")   
    
    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            if i == j:
                continue
            isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            codats_acc = codats(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            without_adapt_acc = without_adapt(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)

            isih_da_house_accs.append(isih_da_house_acc)
            isih_da_season_accs.append(isih_da_season_acc)
            codats_accs.append(codats_acc)
            without_adapt_accs.append(without_adapt_acc)
            patterns.append(f"({i}, s) -> ({j}, w)")
    
    print(f"isih-DA (Household => Season) Average: {sum(isih_da_house_accs)/len(isih_da_house_accs)}")
    print(f"isih-DA (Season => Household) Average: {sum(isih_da_season_accs)/len(isih_da_season_accs)}")
    print(f"CoDATS Average: {sum(codats_accs)/len(codats_accs)}")
    print(f"Without Adapt Average: {sum(without_adapt_accs)/len(without_adapt_accs)}")
    df["PAT"] = patterns
    df["isih-DA (Household => Season)"] = isih_da_house_accs
    df["isih-DA (Season => Household)"] = isih_da_season_accs
    df["CoDATS"] = codats_accs
    df["Wtihout_Adapt"] = without_adapt_accs
    df.to_csv("ecodataset_experiment.csv", index=False)

if __name__ == "__main__":
    main()