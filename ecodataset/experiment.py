import pickle

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from ..utils import utils
from ..models import IsihDanns, Codats, CoDATS_F_C
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDXS = [1, 2, 3, 4, 5]


def isih_da_house(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5, is_kfold_eval: bool=False, num_repeats:int=10) -> torch.Tensor:
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
    scaler.fit(target_X)
    target_X = scaler.transform(target_X)

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

    target_X = scaler.transform(target_X)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)
    with open("isih_dann_tmp.pickle", mode="wb") as f:
        pickle.dump(isih_dann, f)

    if is_kfold_eval:
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
    else:
        accs = []
        for _ in range(num_repeats):
            train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
            source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

            test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
            test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
            test_target_X = test_target_X.to(DEVICE)
            test_target_y_task = test_target_y_task.to(DEVICE)
            ## isih-DA fit, predict for 2nd dimension
            with open("isih_dann_tmp.pickle", mode="rb") as f:
                isih_dann_tmp = pickle.load(f)
            isih_dann_tmp.fit_2nd_dim(source_loader, target_loader, test_target_X, test_target_y_task)
            isih_dann_tmp.set_eval()
            pred_y_task = isih_dann_tmp.predict(test_target_X, is_1st_dim=False)

            # Algo3. Evaluation
            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
        return sum(accs)/num_repeats

def isih_da_season(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5, id_kfold_eval: bool=False, num_repeats:int=10) -> torch.Tensor:
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
    scaler.fit(target_X)
    target_X = scaler.transform(target_X)

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

    target_X = scaler.transform(target_X)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)
    with open("isih_dann_tmp.pickle", mode="wb") as f:
        pickle.dump(isih_dann, f)
    if id_kfold_eval:
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
    else:
        accs = []
        for _ in range(num_repeats):
            train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
            source_loader, target_loader, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

            test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
            test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
            test_target_X = test_target_X.to(DEVICE)
            test_target_y_task = test_target_y_task.to(DEVICE)
            ## isih-DA fit, predict for 2nd dimension
            with open("isih_dann_tmp.pickle", mode="rb") as f:
                isih_dann_tmp = pickle.load(f)
            isih_dann_tmp.fit_2nd_dim(source_loader, target_loader, test_target_X, test_target_y_task)
            isih_dann_tmp.set_eval()
            pred_y_task = isih_dann_tmp.predict(test_target_X, is_1st_dim=False)

            # Algo3. Evaluation
            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
        return sum(accs)/num_repeats


def codats(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5, is_kfold_eval: bool=False, num_repeats:int=10) -> torch.Tensor:
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
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    target_X = scaler.transform(target_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)
    if is_kfold_eval:
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
    else:
        accs = []
        for _ in range(num_repeats):
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
            codats.set_eval()
            pred_y_task = codats.predict(test_target_X)

            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
        return sum(accs)/num_repeats


def without_adapt(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, n_splits: int=5, is_kfold_eval: bool=False, num_repeats:int=10) -> float:
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")

    train_source_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv")[train_source_X.Season==winter_idx].values.reshape(-1)
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)

    train_source_X = train_source_X[train_source_X.Season==winter_idx]
    target_X = target_X[target_X.Season==summer_idx]

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    target_X = scaler.transform(target_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    if is_kfold_eval:
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
            pred_y_task = without_adapt(test_target_X)
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
        return sum(accs)/n_splits
    else:
        accs = []
        for _ in range(num_repeats):
            train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
            source_loader, _, _, _, _, _ = utils.get_loader(train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True)

            test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
            test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
            test_target_X = test_target_X.to(DEVICE)
            test_target_y_task = test_target_y_task.to(DEVICE)

            without_adapt = CoDATS_F_C(input_size=train_source_X.shape[2])
            without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=0.0001)
            criterion = nn.BCELoss()
            without_adapt = utils.fit_without_adaptation(source_loader=source_loader, task_classifier=without_adapt, task_optimizer=without_adapt_optimizer, criterion=criterion, num_epochs=300)
            without_adapt.eval()
            pred_y_task = without_adapt(test_target_X)
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
        return sum(accs)/num_repeats



def train_on_target(target_idx: int, summer_idx: int, n_splits: int=5, is_kfold_eval: bool=False, num_repeats:int=10) -> float:
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[target_X.Season==summer_idx].values.reshape(-1)
    target_X = target_X[target_X.Season==summer_idx]

    scaler = preprocessing.StandardScaler()
    target_X = scaler.fit_transform(target_X)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)
    if is_kfold_eval:
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
                    pred_y_task = train_on_target(target_X_batch)
                    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
                    loss = criterion(pred_y_task, target_y_task_batch)

                    train_on_target_optimizer.zero_grad()
                    loss.backward()
                    train_on_target_optimizer.step()
            pred_y_task = train_on_target(test_target_X)
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
        return sum(accs)/n_splits
    else:
        accs = []
        for _ in range(num_repeats):
            train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(target_X, target_y_task, test_size=0.5, shuffle=False)
            
            
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
                    pred_y_task = train_on_target(target_X_batch)
                    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
                    loss = criterion(pred_y_task, target_y_task_batch)

                    train_on_target_optimizer.zero_grad()
                    loss.backward()
                    train_on_target_optimizer.step()
            train_on_target.eval()
            pred_y_task = train_on_target(test_target_X)
            pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
            pred_y_task = pred_y_task > 0.5
            acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
            accs.append(acc.item())
            ground_truth_ratio = sum(test_target_y_task) / test_target_y_task.shape[0]
        return sum(accs)/num_repeats, ground_truth_ratio.item()

def main():
    isih_da_house_accs = []
    isih_da_season_accs = []
    codats_accs = []
    without_adapt_accs = []
    train_on_target_accs = []
    ground_truth_ratios = []
    df = pd.DataFrame()
    patterns = []

    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            if i == j:
                continue
            elif i == 4:
                j = 5
            elif i == 5:
                j = 4
            elif (i != 4) and (i != 5):
                if (j == 4) or (j == 5):
                    continue
            isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            codats_acc = codats(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            without_adapt_acc = without_adapt(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1)
            train_on_target_acc, ground_truth_ratio = train_on_target(target_idx=j, summer_idx=1)
 
            isih_da_house_accs.append(isih_da_house_acc)
            isih_da_season_accs.append(isih_da_season_acc)
            codats_accs.append(codats_acc)
            without_adapt_accs.append(without_adapt_acc)
            train_on_target_accs.append(train_on_target_acc)
            ground_truth_ratios.append(ground_truth_ratio)
            patterns.append(f"({i}, w) -> ({j}, s)") 
            if (i == 4) or (i == 5):
                break

    for i in HOUSEHOLD_IDXS:
        for j in HOUSEHOLD_IDXS:
            if i == j:
                continue
            elif i == 4:
                j = 5
            elif i == 5:
                j = 4
            elif (i != 4) and (i != 5):
                if (j == 4) or (j == 5):
                    continue
            isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            codats_acc = codats(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            without_adapt_acc = without_adapt(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0)
            train_on_target_acc, ground_truth_ratio = train_on_target(target_idx=j, summer_idx=0)

            isih_da_house_accs.append(isih_da_house_acc)
            isih_da_season_accs.append(isih_da_season_acc)
            codats_accs.append(codats_acc)
            without_adapt_accs.append(without_adapt_acc)
            train_on_target_accs.append(train_on_target_acc)
            ground_truth_ratios.append(ground_truth_ratio)
            patterns.append(f"({i}, s) -> ({j}, w)")
            if (i == 4) or (i == 5):
                break
    
    print(f"isih-DA (Household => Season) Average: {sum(isih_da_house_accs)/len(isih_da_house_accs)}")
    print(f"isih-DA (Season => Household) Average: {sum(isih_da_season_accs)/len(isih_da_season_accs)}")
    print(f"CoDATS Average: {sum(codats_accs)/len(codats_accs)}")
    print(f"Without Adapt Average: {sum(without_adapt_accs)/len(without_adapt_accs)}")
    print(f"Train on Target Average: {sum(train_on_target_accs)/len(train_on_target_accs)}")

    df["PAT"] = patterns
    df["isih-DA (Household => Season)"] = isih_da_house_accs
    df["isih-DA (Season => Household)"] = isih_da_season_accs
    df["CoDATS"] = codats_accs
    df["Wtihout_Adapt"] = without_adapt_accs
    df["Train_on_Target"] = train_on_target_accs
    df["Ground Truth Ratio"] = ground_truth_ratios
    df.to_csv("ecodataset_experiment.csv", index=False)

if __name__ == "__main__":
    main()