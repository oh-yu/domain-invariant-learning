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
HOUSEHOLD_IDXS = [1, 2, 3, 4, 5]
FLAGS = flags.FLAGS
flags.DEFINE_string("algo_name", "DANN", "which algo to be used, DANN or CoRAL")
flags.DEFINE_integer("num_repeats", 10, "the number of evaluation trials")
flags.DEFINE_boolean("is_RV_tuning", True, "Whether or not use Reverse Validation based free params tuning method(5.1.2 algo from DANN paper)")


def _get_source_target_from_ecodataset(source_idx, target_idx, source_season_idx, target_season_idx):
    train_source_X = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv"
    )
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == source_season_idx].values.reshape(-1)
    target_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv"
    )[target_X.Season == target_season_idx].values.reshape(-1)
    train_source_X = train_source_X[train_source_X.Season == source_season_idx]
    target_X = target_X[target_X.Season == target_season_idx]

    scaler = preprocessing.StandardScaler()
    train_source_X = scaler.fit_transform(train_source_X)
    scaler.fit(target_X)
    target_X = scaler.transform(target_X)

    train_source_X, train_source_y_task = utils.apply_sliding_window(
        train_source_X, train_source_y_task, filter_len=6
    )
    target_X, target_y_task = utils.apply_sliding_window(target_X, target_y_task, filter_len=6)

    source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
        train_source_X, target_X, train_source_y_task, target_y_task, shuffle=True, batch_size=32, return_ds=True
    )
    # Note: batch_size=32, because exploding gradient when batch_size=34(this leads to one sample loss)
    return source_loader, target_loader, scaler, source_ds, target_ds, target_X, target_y_task

def _get_target_prime_from_ecodataset(target_prime_idx, target_prime_season_idx):
    target_prime_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_prime_idx}_X_train.csv")
    target_prime_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{target_prime_idx}_Y_train.csv"
    )[target_prime_X.Season == target_prime_season_idx].values.reshape(-1)
    target_prime_X = target_prime_X[target_prime_X.Season == target_prime_season_idx].values

    train_target_prime_X, test_target_prime_X, train_target_prime_y_task, test_target_prime_y_task = train_test_split(
        target_prime_X, target_prime_y_task, test_size=0.5, shuffle=False
    )
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_target_prime_X)
    train_target_prime_X = scaler.transform(train_target_prime_X)
    test_target_prime_X = scaler.transform(test_target_prime_X)
    train_target_prime_X, train_target_prime_y_task = utils.apply_sliding_window(
        train_target_prime_X, train_target_prime_y_task, filter_len=6
    )
    test_target_prime_X, test_target_prime_y_task = utils.apply_sliding_window(test_target_prime_X, test_target_prime_y_task, filter_len=6)
    return train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task


def _get_source_target_prime_from_ecodataset(source_idx, target_prime_idx, source_season_idx, target_prime_season_ix):
    train_source_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_X_train.csv")
    target_prime_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_prime_idx}_X_train.csv")

    train_source_y_task = pd.read_csv(
        f"./domain-invariant-learning/deep_occupancy_detection/data/{source_idx}_Y_train.csv"
    )[train_source_X.Season == source_season_idx].values.reshape(-1)
    target_prime_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_prime_idx}_Y_train.csv")[
        target_prime_X.Season == target_prime_season_ix
    ].values.reshape(-1)

    train_source_X = train_source_X[train_source_X.Season == source_season_idx]
    target_prime_X = target_prime_X[target_prime_X.Season == target_prime_season_ix]

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_source_X)
    train_source_X = scaler.transform(train_source_X)
    train_source_X, train_source_y_task = utils.apply_sliding_window(train_source_X, train_source_y_task, filter_len=6)

    train_target_prime_X, test_target_prime_X, train_target_prime_y_task, test_target_prime_y_task = train_test_split(
        target_prime_X, target_prime_y_task, test_size=0.5, shuffle=False
    )
    scaler.fit(train_target_prime_X)
    train_target_prime_X = scaler.transform(train_target_prime_X)
    test_target_prime_X = scaler.transform(test_target_prime_X)
    train_target_prime_X, train_target_prime_y_task = utils.apply_sliding_window(train_target_prime_X, train_target_prime_y_task, filter_len=6)
    test_target_prime_X, test_target_prime_y_task = utils.apply_sliding_window(test_target_prime_X, test_target_prime_y_task, filter_len=6)
    return train_source_X, train_source_y_task, train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task

def danns_2d(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, num_repeats: int = 10,) -> float:
    accs = []
    for _ in range(num_repeats):
        # Prepare Data
        source_loader, target_loader, scaler, _, _, _, _ = _get_source_target_from_ecodataset(source_idx=source_idx, target_idx=target_idx, source_season_idx=winter_idx, target_season_idx=winter_idx)
        train_target_prime_X, _, test_target_prime_X, test_target_prime_y_task = _get_target_prime_from_ecodataset(target_prime_idx=target_idx, target_prime_season_idx=summer_idx)

        test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32)
        test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.float32)
        test_target_prime_X = test_target_prime_X.to(DEVICE)
        test_target_prime_y_task = test_target_prime_y_task.to(DEVICE)

        train_target_prime_X = torch.tensor(train_target_prime_X, dtype=torch.float32).to(DEVICE)
        train_target_prime_y_domain = torch.ones(train_target_prime_X.shape[0]).to(DEVICE)
        target_prime_ds = TensorDataset(train_target_prime_X, train_target_prime_y_domain)
        target_prime_loader = DataLoader(target_prime_ds, shuffle=True)

        # Init 2D-DANNs
        danns_2d = Danns2D(experiment="ECOdataset")
        acc = danns_2d.fit(
            source_loader,
            target_loader,
            target_prime_loader,
            test_target_prime_X,
            test_target_prime_y_task
        )
        accs.append(acc)
    return sum(accs) / num_repeats


def isih_da_house(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, num_repeats: int = 10,) -> float:
    """
    Execute isih-DA (Household => Season) experiment.
    TODO: Attach paper
    """
    accs = []
    for _ in range(num_repeats):
        # Algo1. Inter-Households DA
        ## Prepare Data
        _, _, scaler, source_ds, target_ds, test_target_X, test_target_y_task = _get_source_target_from_ecodataset(source_idx=source_idx,
                                                                                                                  target_idx=target_idx,
                                                                                                                  source_season_idx=winter_idx,
                                                                                                                  target_season_idx=winter_idx
                                                                                                                  )
        target_X = test_target_X

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        ## isih-DA fit, predict for 1st dimension
        isih_dann = IsihDanns(experiment="ECOdataset")
        isih_dann.fit_1st_dim(source_ds, target_ds, test_target_X, test_target_y_task)
        pred_y_task = isih_dann.predict_proba(test_target_X, is_1st_dim=True)

        # Algo2. Inter-Seasons DA
        ## Prepare Data
        train_source_X = target_X
        train_source_y_task = pred_y_task.cpu().detach().numpy()
        train_target_X, train_target_y_task, test_target_X, test_target_y_task = _get_target_prime_from_ecodataset(target_prime_idx=target_idx, target_prime_season_idx=summer_idx)

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


def isih_da_season(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, num_repeats: int = 10,) -> float:
    """
    Execute isih-DA (Season => Household) experiment.
    TODO: Attach paper
    """
    accs = []
    for _ in range(num_repeats):
        # Algo1. Inter-Seasons DA
        ## Prepare Data
        _, _, scaler, source_ds, target_ds, test_target_X, test_target_y_task = _get_source_target_from_ecodataset(source_idx=source_idx, target_idx=source_idx, source_season_idx=winter_idx, target_season_idx=summer_idx)
        target_X = test_target_X
        
        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        ## isih-DA fit, predict for 1st dimension
        isih_dann = IsihDanns(experiment="ECOdataset")
        isih_dann.fit_1st_dim(source_ds, target_ds, test_target_X, test_target_y_task)
        pred_y_task = isih_dann.predict_proba(test_target_X, is_1st_dim=True)

        # Algo2. Inter-Households DA
        ## Prepare Data
        train_source_X = target_X
        train_source_y_task = pred_y_task.cpu().detach().numpy()
        train_target_X, train_target_y_task, test_target_X, test_target_y_task = _get_target_prime_from_ecodataset(target_prime_idx=target_idx, target_prime_season_idx=summer_idx)
        source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True, return_ds=True
        )

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)
        ## isih-DA fit, predict for 2nd dimension
        isih_dann.fit_2nd_dim(source_ds, target_ds, test_target_X, test_target_y_task)
        isih_dann.set_eval()
        pred_y_task = isih_dann.predict(test_target_X, is_1st_dim=False)

        # Algo3. Evaluation
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def codats(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, num_repeats: int = 10,) -> float:
    """
    Execute CoDATS experiment.
    TODO: Attach paper
    """
    # Direct Inter-Seasons and Inter-Households DA
    ## Prepare Data
    train_source_X, train_source_y_task, train_target_X, train_target_y_task, test_target_X, test_target_y_task = _get_source_target_prime_from_ecodataset(
        source_idx=source_idx,
        target_prime_idx=target_idx,
        source_season_idx=winter_idx,
        target_prime_season_ix=summer_idx
    )
    accs = []
    for _ in range(num_repeats):
        source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True, return_ds=True
        )

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        ## CoDATS fit, predict
        codats = Codats(experiment="ECOdataset")
        acc = codats.fit(source_ds, target_ds, test_target_X, test_target_y_task)
        accs.append(acc)
    return sum(accs) / num_repeats


def without_adapt(source_idx: int, target_idx: int, winter_idx: int, summer_idx: int, num_repeats: int = 10,) -> float:
    train_source_X, train_source_y_task, train_target_X, train_target_y_task, test_target_X, test_target_y_task = _get_source_target_prime_from_ecodataset(
        source_idx=source_idx,
        target_prime_idx=target_idx,
        source_season_idx=winter_idx,
        target_prime_season_ix=summer_idx
    )
    accs = []
    for _ in range(num_repeats):

        source_loader, _, _, _, _, _ = utils.get_loader(
            train_source_X, train_target_X, train_source_y_task, train_target_y_task, shuffle=True
        )

        test_target_X = torch.tensor(test_target_X, dtype=torch.float32)
        test_target_y_task = torch.tensor(test_target_y_task, dtype=torch.float32)
        test_target_X = test_target_X.to(DEVICE)
        test_target_y_task = test_target_y_task.to(DEVICE)

        without_adapt = CoDATS_F_C(experiment="ECOdataset")
        without_adapt.fit_without_adapt(source_loader)
        without_adapt.eval()
        pred_y_task = without_adapt(test_target_X)
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
    return sum(accs) / num_repeats


def train_on_target(target_idx: int, summer_idx: int, num_repeats: int = 10) -> float:
    target_X = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_X_train.csv")
    target_y_task = pd.read_csv(f"./domain-invariant-learning/deep_occupancy_detection/data/{target_idx}_Y_train.csv")[
        target_X.Season == summer_idx
    ].values.reshape(-1)
    target_X = target_X[target_X.Season == summer_idx]

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
        train_on_target = CoDATS_F_C(experiment="ECOdataset")
        train_on_target.fit_on_target(target_loader)

        train_on_target.eval()
        pred_y_task = train_on_target(test_target_X)
        pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
        pred_y_task = pred_y_task > 0.5
        acc = sum(pred_y_task == test_target_y_task) / test_target_y_task.shape[0]
        accs.append(acc.item())
        ground_truth_ratio = sum(test_target_y_task) / test_target_y_task.shape[0]
    return sum(accs) / num_repeats, ground_truth_ratio.item()


def main(argv):
    danns_2d_accs = []
    isih_da_house_accs = []
    isih_da_season_accs = []
    codats_accs = []
    without_adapt_accs = []
    train_on_target_accs = []
    ground_truth_ratios = []
    df = pd.DataFrame()
    patterns = []

    for i in tqdm(HOUSEHOLD_IDXS):
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
            danns_2d_acc = danns_2d(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1, num_repeats=FLAGS.num_repeats)
            isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1, num_repeats=FLAGS.num_repeats)
            isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1, num_repeats=FLAGS.num_repeats)
            codats_acc = codats(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1, num_repeats=FLAGS.num_repeats)
            without_adapt_acc = without_adapt(source_idx=i, target_idx=j, winter_idx=0, summer_idx=1, num_repeats=FLAGS.num_repeats)
            train_on_target_acc, ground_truth_ratio = train_on_target(target_idx=j, summer_idx=1, num_repeats=FLAGS.num_repeats)

            danns_2d_accs.append(danns_2d_acc)
            isih_da_house_accs.append(isih_da_house_acc)
            isih_da_season_accs.append(isih_da_season_acc)
            codats_accs.append(codats_acc)
            without_adapt_accs.append(without_adapt_acc)
            train_on_target_accs.append(train_on_target_acc)
            ground_truth_ratios.append(ground_truth_ratio)
            patterns.append(f"({i}, w) -> ({j}, s)")
            if (i == 4) or (i == 5):
                break

    for i in tqdm(HOUSEHOLD_IDXS):
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
            danns_2d_acc = danns_2d(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0, num_repeats=FLAGS.num_repeats)
            isih_da_house_acc = isih_da_house(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0, num_repeats=FLAGS.num_repeats)
            isih_da_season_acc = isih_da_season(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0, num_repeats=FLAGS.num_repeats)
            codats_acc = codats(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0, num_repeats=FLAGS.num_repeats)
            without_adapt_acc = without_adapt(source_idx=i, target_idx=j, winter_idx=1, summer_idx=0, num_repeats=FLAGS.num_repeats)
            train_on_target_acc, ground_truth_ratio = train_on_target(target_idx=j, summer_idx=0, num_repeats=FLAGS.num_repeats)

            danns_2d_accs.append(danns_2d_acc)
            isih_da_house_accs.append(isih_da_house_acc)
            isih_da_season_accs.append(isih_da_season_acc)
            codats_accs.append(codats_acc)
            without_adapt_accs.append(without_adapt_acc)
            train_on_target_accs.append(train_on_target_acc)
            ground_truth_ratios.append(ground_truth_ratio)
            patterns.append(f"({i}, s) -> ({j}, w)")
            if (i == 4) or (i == 5):
                break
    print(f"DANNs-2D Average: {sum(danns_2d_accs) / len(danns_2d_accs)}")
    print(f"isih-DA (Household => Season) Average: {sum(isih_da_house_accs)/len(isih_da_house_accs)}")
    print(f"isih-DA (Season => Household) Average: {sum(isih_da_season_accs)/len(isih_da_season_accs)}")
    print(f"CoDATS Average: {sum(codats_accs)/len(codats_accs)}")
    print(f"Without Adapt Average: {sum(without_adapt_accs)/len(without_adapt_accs)}")
    print(f"Train on Target Average: {sum(train_on_target_accs)/len(train_on_target_accs)}")

    df["PAT"] = patterns
    df["DANNs-2D"] = danns_2d_accs
    df["isih-DA (Household => Season)"] = isih_da_house_accs
    df["isih-DA (Season => Household)"] = isih_da_season_accs
    df["CoDATS"] = codats_accs
    df["Wtihout_Adapt"] = without_adapt_accs
    df["Train_on_Target"] = train_on_target_accs
    df["Ground Truth Ratio"] = ground_truth_ratios
    df.to_csv(f"ecodataset_{str(datetime.now())}_{FLAGS.algo_name}.csv", index=False)


if __name__ == "__main__":
    app.run(main)
