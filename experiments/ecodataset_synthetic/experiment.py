from datetime import datetime

import pandas as pd
import torch
from absl import app, flags
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ...networks import Codats, CoDATS_F_C, IsihDanns, Danns2D
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
flags.DEFINE_string("algo_name", "DANN", "which algo to be used, DANN or CoRAL")
flags.DEFINE_integer("num_repeats", 10 , "the number of evaluation trials")
flags.DEFINE_boolean("is_RV_tuning", True, "Whether or not use Reverse Validation based free params tuning method(5.1.2 algo from DANN paper)")

flags.mark_flag_as_required("lag_1")
flags.mark_flag_as_required("lag_2")


def danns_2d(source_idx=2, season_idx=0, num_repeats: int = 10):
    accs = []
    for _ in range(num_repeats):
        train_source_X = pd.read_csv(
            f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv"
        )
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

        train_source_X, train_source_y_task = utils.apply_sliding_window(
            train_source_X, train_source_y_task, filter_len=6
        )
        target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)
        source_loader, target_loader, _, _, _, _  = utils.get_loader(
            train_source_X, target_X, train_source_y_task, target_y_task, shuffle=True, batch_size=32
        )

        train_target_prime_X, test_target_prime_X, train_target_prime_y_task, test_target_prime_y_task = train_test_split(
            target_prime_X, target_prime_y_task, test_size=0.5, shuffle=False
        )
        scaler.fit(train_target_prime_X)
        train_target_prime_X = scaler.transform(train_target_prime_X)
        test_target_prime_X = scaler.transform(test_target_prime_X)
        train_target_prime_X, train_target_prime_y_task = utils.apply_sliding_window(
            train_target_prime_X, train_target_prime_y_task, filter_len=6
        )
        test_target_prime_X, test_target_prime_y_task = utils.apply_sliding_window(test_target_prime_X, test_target_prime_y_task, filter_len=6)
        test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32)
        test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.float32)
        test_target_prime_X = test_target_prime_X.to(DEVICE)
        test_target_prime_y_task = test_target_prime_y_task.to(DEVICE)

        train_target_prime_X = torch.tensor(train_target_prime_X, dtype=torch.float32).to(DEVICE)
        train_target_prime_y_domain = torch.ones(train_target_prime_X.shape[0]).to(DEVICE)
        target_prime_ds = TensorDataset(train_target_prime_X, train_target_prime_y_domain)
        target_prime_loader = DataLoader(target_prime_ds, shuffle=True)

        # 2D-DANNs
        danns_2d = Danns2D(experiment="ECOdataset_synthetic")
        acc = danns_2d.fit(
            source_loader,
            target_loader,
            target_prime_loader,
            test_target_prime_X,
            test_target_prime_y_task
        )
        accs.append(acc)
    return sum(accs) / num_repeats



def isih_da(source_idx=2, season_idx=0, num_repeats: int = 10):
    accs = []
    for _ in range(num_repeats):
        train_source_X = pd.read_csv(
            f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv"
        )
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

        train_source_X, train_source_y_task = utils.apply_sliding_window(
            train_source_X, train_source_y_task, filter_len=6
        )
        target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

        train_target_X, test_target_X, train_target_y_task, test_target_y_task = (
            target_X,
            target_X,
            target_y_task,
            target_y_task,
        )
        source_loader, target_loader, train_source_y_task, train_source_X, _, _, source_ds, target_ds = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True, batch_size=32, return_ds=True
        )
        # Note: batch_size=32, because exploding gradient when batch_size=34(this leads to one sample loss)

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        isih_dann = IsihDanns(experiment="ECOdataset_synthetic")
        isih_dann.fit_1st_dim(source_ds, target_ds, test_target_X, test_target_y_task)
        pred_y_task = isih_dann.predict_proba(test_target_X, is_1st_dim=True)
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
        test_target_X, test_target_y_task = utils.apply_sliding_window(test_target_X, test_target_y_task, filter_len=6)
        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)
        source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True, return_ds=True
        )
        ## isih-DA fit, predict for 2nd dimension
        isih_dann.fit_2nd_dim(source_ds, target_ds, test_target_X, test_target_y_task)
        isih_dann.set_eval()
        pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=False)

        # Algo3. Evaluation
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def codats(source_idx=2, season_idx=0, num_repeats: int = 10):
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
    train_target_X, train_target_y_task = utils.apply_sliding_window(train_target_X, train_target_y_task, filter_len=6)
    test_target_X, test_target_y_task = utils.apply_sliding_window(test_target_X, test_target_y_task, filter_len=6)
    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)
    for _ in range(num_repeats):
        source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True, return_ds=True
        )
        ## CoDATS fit, predict
        codats = Codats(experiment="ECOdataset_synthetic")
        acc = codats.fit(source_ds, target_ds, test_target_X, test_target_y_task)
        accs.append(acc)
    return sum(accs) / num_repeats


def without_adapt(source_idx=2, season_idx=0, num_repeats: int = 10):
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
    train_target_X, train_target_y_task = utils.apply_sliding_window(train_target_X, train_target_y_task, filter_len=6)
    test_target_X, test_target_y_task = utils.apply_sliding_window(test_target_X, test_target_y_task, filter_len=6)
    test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
    test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
    test_target_X = test_target_X.to(DEVICE)
    test_target_y_task = test_target_y_task.to(DEVICE)
    for _ in range(num_repeats):
        source_loader, _, _, _, _, _ = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True
        )
        ## Without Adapt fit, predict
        without_adapt = CoDATS_F_C(experiment="ECOdataset_synthetic")
        without_adapt.fit_without_adapt(source_loader)

        without_adapt.eval()
        pred_y = without_adapt(test_target_X)
        pred_y = torch.sigmoid(pred_y).reshape(-1)
        pred_y = pred_y > 0.5
        acc = sum(pred_y == test_target_y_task) / pred_y.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def train_on_target(source_idx=2, season_idx=0, num_repeats: int = 10):
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
        train_on_target = CoDATS_F_C(experiment="ECOdataset_synthetic")
        train_on_target.fit_on_target(target_loader)
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
    accs_danns_2d = []
    accs_isih_da = []
    accs_codats = []
    accs_without_adapt = []
    accs_train_on_target = []
    patterns = []
    for i in tqdm(HOUSEHOLD_IDX):
        for j in SEASON_IDX:
            acc_danns_2d = danns_2d(source_idx=i, season_idx=j, num_repeats=FLAGS.num_repeats)
            acc_isih_da = isih_da(source_idx=i, season_idx=j, num_repeats=FLAGS.num_repeats)
            acc_codats = codats(source_idx=i, season_idx=j, num_repeats=FLAGS.num_repeats)
            acc_without_adapt = without_adapt(source_idx=i, season_idx=j, num_repeats=FLAGS.num_repeats)
            acc_train_on_target = train_on_target(source_idx=i, season_idx=j, num_repeats=FLAGS.num_repeats)
            accs_danns_2d.append(acc_danns_2d)
            accs_isih_da.append(acc_isih_da)
            accs_codats.append(acc_codats)
            accs_without_adapt.append(acc_without_adapt)
            accs_train_on_target.append(acc_train_on_target)
            patterns.append(f"Household ID:{i}, Season:{j}")
    df = pd.DataFrame()
    df["patterns"] = patterns
    df["accs_danns_2d"] = accs_danns_2d
    df["accs_isih_da"] = accs_isih_da
    df["accs_codats"] = accs_codats
    df["accs_without_adapt"] = accs_without_adapt
    df["accs_train_on_target"] = accs_train_on_target
    df.to_csv(
        f"ecodataset_synthetic_lag{FLAGS.lag_1}_lag{FLAGS.lag_2}_{str(datetime.now())}_{FLAGS.algo_name}.csv",
        index=False,
    )


if __name__ == "__main__":
    app.run(main)
