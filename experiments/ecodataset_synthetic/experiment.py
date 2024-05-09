import pickle

import pandas as pd
import torch
from absl import app, flags
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ...networks import Codats, CoDATS_F_C, IsihDanns
from ...utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOUSEHOLD_IDX = [1, 2, 3, 4, 5]
SEASON_IDX = [0, 1]
LAG_NUM_TO_TIME_LIST = {
    1: [i for i in range(13, 44, 1)] + [12],
    2: [i for i in range(14, 44, 1)] + [12, 13],
    3: [i for i in range(15, 44, 1)] + [12, 13, 14],
    4: [i for i in range(16, 44, 1)] + [12, 13, 14, 15],
    5: [i for i in range(17, 44, 1)] + [12, 13, 14, 15, 16],
    6: [i for i in range(18, 44, 1)] + [12, 13, 14, 15, 16, 17],
}
FLAGS = flags.FLAGS
flags.DEFINE_integer("lag_1", 1, "time lag for intermediate domain")
flags.DEFINE_integer("lag_2", 6, "time lag for terminal domain")
flags.DEFINE_boolean("is_kfold_eval", False, "use kfold cross validation,  otherwise use train-test split. currently isih-DA is not supported.")
flags.mark_flag_as_required("lag_1")
flags.mark_flag_as_required("lag_2")


def isih_da(source_idx=2, season_idx=0, num_repeats: int = 10):
    accs = []
    for _ in range(num_repeats):
        train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
        train_source_y_task = pd.read_csv(
            f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
        )[train_source_X.Season == season_idx]
        train_source_X = train_source_X[train_source_X.Season == season_idx]

        target_X = train_source_X.copy()
        time_list = LAG_NUM_TO_TIME_LIST[FLAGS.lag_1]
        time_list = time_list * int(train_source_X.shape[0] / 32)
        target_X["Time"] = time_list
        target_y_task = train_source_y_task

        target_prime_X = train_source_X.copy()
        time_list = LAG_NUM_TO_TIME_LIST[FLAGS.lag_2]
        time_list = time_list * int(train_source_X.shape[0] / 32)
        target_prime_X["Time"] = time_list
        target_prime_y_task = train_source_y_task

        train_source_y_task = train_source_y_task.values.reshape(-1)
        target_y_task = target_y_task.values.reshape(-1)
        target_prime_y_task = target_prime_y_task.values.reshape(-1)

        scaler = preprocessing.StandardScaler()
        train_source_X = scaler.fit_transform(train_source_X)
        scaler.fit(target_X)
        target_X = scaler.transform(target_X)

        train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)
        target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

        train_target_X, test_target_X, train_target_y_task, test_target_y_task = (
            target_X,
            target_X,
            target_y_task,
            target_y_task,
        )
        source_loader, target_loader, train_source_y_task, train_source_X, _, _ = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True
        )

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        isih_dann = IsihDanns(
            input_size=train_source_X.shape[2],
            hidden_size=128,
            lr_dim1=0.0001,
            lr_dim2=0.00005,
            num_epochs_dim1=200,
            num_epochs_dim2=100,
            experiment="ECOdataset_synthetic"
        )
        isih_dann.fit_1st_dim(source_loader, target_loader, test_target_X, test_target_y_task)
        pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=True)
        pred_y_task = pred_y_task > 0.5
        train_source_X = target_X
        train_source_y_task = pred_y_task.cpu().detach().numpy()
        target_X = target_prime_X.values
        target_y_task = target_prime_y_task

        
        train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(
            target_X, target_y_task, test_size=0.5, shuffle=False
        )
        scaler.fit(train_target_X)
        train_target_X = scaler.transform(train_target_X)
        test_target_X = scaler.transform(test_target_X)
        train_target_X, train_target_y_task = utils.apply_sliding_window(
            train_target_X, train_target_y_task, filter_len=6
        )
        test_target_X, test_target_y_task = utils.apply_sliding_window(
            test_target_X, test_target_y_task, filter_len=6
        )
        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)
        source_loader, target_loader, _, _, _, _ = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True
        )
        ## isih-DA fit, predict for 2nd dimension
        isih_dann.fit_2nd_dim(source_loader, target_loader, test_target_X, test_target_y_task)
        isih_dann.set_eval()
        pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=False)

        # Algo3. Evaluation
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def codats(source_idx=2, season_idx=0, n_splits: int = 5, is_kfold_eval: bool = False, num_repeats: int = 10):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_prime_X = train_source_X.copy()
    time_list = LAG_NUM_TO_TIME_LIST[FLAGS.lag_2]
    time_list = time_list * int(train_source_X.shape[0] / 32)
    target_prime_X["Time"] = time_list
    target_prime_y_task = train_source_y_task

    train_source_y_task = train_source_y_task.values.reshape(-1)
    target_prime_y_task = target_prime_y_task.values.reshape(-1)

    target_X, target_y_task = target_prime_X.values, target_prime_y_task
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)

    accs = []
    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(
        target_X, target_y_task, test_size=0.5, shuffle=False
    )
    scaler.fit(train_target_X)
    train_target_X = scaler.transform(train_target_X)
    test_target_X = scaler.transform(test_target_X)
    train_target_X, train_target_y_task = utils.apply_sliding_window(
        train_target_X, train_target_y_task, filter_len=6
    )
    test_target_X, test_target_y_task = utils.apply_sliding_window(
        test_target_X, test_target_y_task, filter_len=6
    )
    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)
    for _ in range(num_repeats):
        source_loader, target_loader, _, _, _, _ = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True
        )
        ## CoDATS fit, predict
        codats = Codats(input_size=train_source_X.shape[2], hidden_size=128, lr=0.0001, num_epochs=300)
        codats.fit(source_loader, target_loader, test_target_X, test_target_y_task)
        codats.set_eval()
        pred_y_task = codats.predict(test_target_X)

        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def without_adapt(source_idx=2, season_idx=0, n_splits: int = 5, is_kfold_eval: bool = False, num_repeats: int = 10):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_prime_X = train_source_X.copy()
    time_list = LAG_NUM_TO_TIME_LIST[FLAGS.lag_2]
    time_list = time_list * int(train_source_X.shape[0] / 32)
    target_prime_X["Time"] = time_list
    target_prime_y_task = train_source_y_task

    train_source_y_task = train_source_y_task.values.reshape(-1)
    target_prime_y_task = target_prime_y_task.values.reshape(-1)

    target_X, target_y_task = target_prime_X.values, target_prime_y_task
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)

    accs = []
    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(
        target_X, target_y_task, test_size=0.5, shuffle=False
    )
    scaler.fit(train_target_X)
    train_target_X = scaler.transform(train_target_X)
    test_target_X = scaler.transform(test_target_X)
    train_target_X, train_target_y_task = utils.apply_sliding_window(
        train_target_X, train_target_y_task, filter_len=6
    )
    test_target_X, test_target_y_task = utils.apply_sliding_window(
        test_target_X, test_target_y_task, filter_len=6
    )
    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)
    for _ in range(num_repeats):
        source_loader, _, _, _, _, _ = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True
        )
        ## Without Adapt fit, predict
        without_adapt = CoDATS_F_C(input_size=train_source_X.shape[2], experiment="ECOdataset_synthetic")
        without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        without_adapt = utils.fit_without_adaptation(
            source_loader=source_loader,
            task_classifier=without_adapt,
            task_optimizer=without_adapt_optimizer,
            criterion=criterion,
            num_epochs=300,
        )
        without_adapt.eval()
        pred_y = without_adapt(test_target_X)
        pred_y = torch.sigmoid(pred_y).reshape(-1)
        pred_y = pred_y > 0.5
        acc = sum(pred_y == test_target_y_task) / pred_y.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def train_on_target(source_idx=2, season_idx=0, n_splits: int = 5, is_kfold_eval: bool = False, num_repeats: int = 10):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == season_idx]
    train_source_X = train_source_X[train_source_X.Season == season_idx]

    target_prime_X = train_source_X.copy()
    time_list = LAG_NUM_TO_TIME_LIST[FLAGS.lag_2]
    time_list = time_list * int(train_source_X.shape[0] / 32)
    target_prime_X["Time"] = time_list
    target_prime_y_task = train_source_y_task
    target_prime_y_task = target_prime_y_task.values.reshape(-1)

    target_X, target_y_task = target_prime_X, target_prime_y_task
    scaler = preprocessing.StandardScaler()
    target_X = scaler.fit_transform(target_X)
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    accs = []
    train_target_X, test_target_X, train_target_y_task, test_target_y_task = train_test_split(
        target_X, target_y_task, test_size=0.5, shuffle=False
    )

    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)

    train_target_X = torch.tensor(train_target_X, dtype=torch.float32)
    train_target_y_task = torch.tensor(train_target_y_task, dtype=torch.float32)
    train_target_X = train_target_X.to(DEVICE)
    train_target_y_task = train_target_y_task.to(DEVICE)
    target_ds = TensorDataset(train_target_X, train_target_y_task)
    for _ in range(num_repeats):
        target_loader = DataLoader(target_ds, batch_size=32, shuffle=True)
        ## Train on Target fit, predict
        train_on_target = CoDATS_F_C(input_size=train_target_X.shape[2], experiment="ECOdataset_synthetic")
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
        train_on_target.eval()
        pred_y_task = train_on_target(test_target_X)
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / pred_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def main(argv):
    assert FLAGS.lag_1 in [1, 2, 3, 4, 5, 6]
    assert (FLAGS.lag_2 in [1, 2, 3, 4, 5, 6]) and (FLAGS.lag_2 > FLAGS.lag_1)
    accs_isih_da = []
    accs_codats = []
    accs_without_adapt = []
    accs_train_on_target = []
    patterns = []
    for i in tqdm(HOUSEHOLD_IDX):
        for j in SEASON_IDX:
            acc_isih_da = isih_da(source_idx=i, season_idx=j)
            acc_codats = codats(source_idx=i, season_idx=j, is_kfold_eval=FLAGS.is_kfold_eval)
            acc_without_adapt = without_adapt(source_idx=i, season_idx=j, is_kfold_eval=FLAGS.is_kfold_eval)
            acc_train_on_target = train_on_target(source_idx=i, season_idx=j, is_kfold_eval=FLAGS.is_kfold_eval)
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
    df.to_csv(f"ecodataset_synthetic_experiment_lag{FLAGS.lag_1}_lag{FLAGS.lag_2}.csv", index=False)


if __name__ == "__main__":
    app.run(main)
