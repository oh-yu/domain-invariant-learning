from datetime import datetime

import pandas as pd
import torch
from absl import app, flags
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ...networks import Codats, CoDATS_F_C, Danns2D, IsihDanns, Rdann
from ...utils import utils

GT_TO_INT = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4, "sit": 5}
USER_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
MODEL_LIST = ["nexus4", "s3", "samsungold", "s3mini"]
ACCELEROMETER_DF = pd.read_csv(
    "./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv"
)
ACCELEROMETER_DF = ACCELEROMETER_DF.add_suffix("_accele")
GYROSCOPE_DF = pd.read_csv(
    "./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv"
)
GYROSCOPE_DF = GYROSCOPE_DF.add_suffix("_gyro")
FLAGS = flags.FLAGS
flags.DEFINE_string("algo_name", "DANN", "which algo to be used, DANN or CoRAL")
flags.DEFINE_integer("num_repeats", 10, "the number of repetitions for hold-out test")
flags.DEFINE_boolean(
    "is_RV_tuning",
    True,
    "Whether or not use Reverse Validation based free params tuning method(5.1.2 algo from DANN paper)",
)


class Pattern:
    def __init__(self, source_user, source_model, target_user, target_model):
        assert source_user in USER_LIST
        assert target_user in USER_LIST
        assert source_model in MODEL_LIST
        assert target_model in MODEL_LIST
        self.source_user = source_user
        self.source_model = source_model
        self.target_user = target_user
        self.target_model = target_model


def get_data_for_uda(user, model, is_targer_prime: bool = False):
    assert model in MODEL_LIST
    assert user in USER_LIST
    df = pd.merge(
        ACCELEROMETER_DF,
        GYROSCOPE_DF,
        left_on=["Arrival_Time_accele", "User_accele", "Device_accele"],
        right_on=["Arrival_Time_gyro", "User_gyro", "Device_gyro"],
        how="left",
    )
    df[["x_gyro", "y_gyro", "z_gyro"]] = df[["x_gyro", "y_gyro", "z_gyro"]].interpolate()
    df[["x_accele", "y_accele", "z_accele"]] = df[["x_accele", "y_accele", "z_accele"]].interpolate()
    df = df[df.User_accele == user]
    df = df[df.Model_accele == model]
    df = df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro", "gt_accele"]].dropna(how="any")
    df = df.reset_index()
    df["gt_accele"] = df["gt_accele"].apply(lambda x: GT_TO_INT[x])
    scaler = StandardScaler()
    if not is_targer_prime:
        df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]] = scaler.fit_transform(
            df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]]
        )
        X, y = utils.apply_sliding_window(
            df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values,
            df["gt_accele"].values.reshape(-1),
            filter_len=128,
            is_overlap=False,
        )
        return X, y
    else:
        X, y = utils.apply_sliding_window(
            df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values,
            df["gt_accele"].values.reshape(-1),
            filter_len=128,
            is_overlap=False,
        )
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, stratify=y)
        train_N, T, H = train_X.shape
        test_N = test_X.shape[0]
        train_X, test_X = train_X.reshape(train_N * T, H), test_X.reshape(test_N * T, H)
        train_X, test_X = scaler.fit_transform(train_X), scaler.transform(test_X)
        train_X, test_X = train_X.reshape(train_N, T, H), test_X.reshape(test_N, T, H)
        return train_X, train_y, test_X, test_y


def danns_2d(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    target_X, target_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(
        user=pattern.target_user, model=pattern.target_model, is_targer_prime=True
    )
    source_loader, target_loader, _, _, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True
    )
    train_target_prime_X = torch.tensor(train_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    train_target_prime_y_domain = torch.ones(train_target_prime_X.shape[0]).to(utils.DEVICE)
    train_tartget_prime_ds = TensorDataset(train_target_prime_X, train_target_prime_y_domain)
    target_prime_loader = DataLoader(train_tartget_prime_ds, shuffle=True, batch_size=128)

    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task).to(utils.DEVICE)
    # 2d DANNs
    danns_2d = Danns2D(experiment="HHAR")
    acc = danns_2d.fit(source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task)
    return acc


def isih_da_user(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    target_X, target_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(
        user=pattern.target_user, model=pattern.target_model, is_targer_prime=True
    )

    # Algo1: Inter-user DA
    source_loader, target_loader, _, _, target_X, target_y_task, source_ds, target_ds = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True, return_ds=True
    )
    isih_dann = IsihDanns(experiment="HHAR")
    isih_dann.fit_1st_dim(source_ds, target_ds, target_X, target_y_task)
    pred_y_task = isih_dann.predict_proba(target_X, is_1st_dim=True)

    # Algo2: Inter-models DA
    source_X = target_X.cpu().detach().numpy()
    source_y_task = pred_y_task.cpu().detach().numpy()
    source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
        source_X,
        train_target_prime_X,
        source_y_task,
        train_target_prime_y_task,
        batch_size=128,
        shuffle=True,
        return_ds=True,
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    isih_dann.fit_2nd_dim(source_ds, target_ds, test_target_prime_X, test_target_prime_y_task)
    isih_dann.set_eval()
    pred_y_task = isih_dann.predict(test_target_prime_X, is_1st_dim=False)

    # Algo3: Evaluation
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def isih_da_model(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    target_X, target_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.target_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(
        user=pattern.target_user, model=pattern.target_model, is_targer_prime=True
    )

    # Algo1: Inter-models DA
    _, _, _, _, target_X, target_y_task, source_ds, target_ds = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True, return_ds=True
    )
    isih_dann = IsihDanns(experiment="HHAR")
    isih_dann.fit_1st_dim(source_ds, target_ds, target_X, target_y_task)
    pred_y_task = isih_dann.predict_proba(target_X, is_1st_dim=True)

    # Algo2: Inter-users DA
    source_X = target_X.cpu().detach().numpy()
    source_y_task = pred_y_task.cpu().detach().numpy()
    source_loader, target_loader, _, _, _, _, source_ds, target_ds = utils.get_loader(
        source_X,
        train_target_prime_X,
        source_y_task,
        train_target_prime_y_task,
        batch_size=128,
        shuffle=True,
        return_ds=True,
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    isih_dann.fit_2nd_dim(source_ds, target_ds, test_target_prime_X, test_target_prime_y_task)
    isih_dann.set_eval()
    pred_y_task = isih_dann.predict(test_target_prime_X, is_1st_dim=False)

    # Algo3: Evaluation
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def codats(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(
        user=pattern.target_user, model=pattern.target_model, is_targer_prime=True
    )

    # Direct Inter-Users and Inter-models DA
    (
        source_loader,
        target_loader,
        _,
        _,
        train_target_prime_X,
        train_target_prime_y_task,
        source_ds,
        target_ds,
    ) = utils.get_loader(
        source_X,
        train_target_prime_X,
        source_y_task,
        train_target_prime_y_task,
        batch_size=128,
        shuffle=True,
        return_ds=True,
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.float32)
    test_target_prime_X = test_target_prime_X.to(utils.DEVICE)
    test_target_prime_y_task = test_target_prime_y_task.to(utils.DEVICE)

    codats = Codats(experiment="HHAR")
    # codats = Rdann(experiment="HHAR")
    acc = codats.fit(source_ds, target_ds, test_target_prime_X, test_target_prime_y_task)
    return acc


def without_adapt(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(
        user=pattern.target_user, model=pattern.target_model, is_targer_prime=True
    )

    # Without Adapt
    source_loader, _, _, _, _, _ = utils.get_loader(
        source_X, train_target_prime_X, source_y_task, train_target_prime_y_task, batch_size=128, shuffle=True
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.float32)
    test_target_prime_X = test_target_prime_X.to(utils.DEVICE)
    test_target_prime_y_task = test_target_prime_y_task.to(utils.DEVICE)

    without_adapt = CoDATS_F_C(experiment="HHAR")
    without_adapt.fit_without_adapt(source_loader)
    without_adapt.eval()

    pred_y_task = without_adapt.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def train_on_target(pattern):
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(
        user=pattern.target_user, model=pattern.target_model, is_targer_prime=True
    )

    train_target_prime_X = torch.tensor(train_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    train_target_prime_y_task = torch.tensor(train_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)

    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    target_prime_ds = TensorDataset(train_target_prime_X, train_target_prime_y_task)
    target_prime_loader = DataLoader(target_prime_ds, batch_size=128, shuffle=True)

    train_on_target = CoDATS_F_C(experiment="HHAR")
    train_on_target.fit_on_target(target_prime_loader)
    train_on_target.eval()
    pred_y_task = train_on_target.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def main(argv):
    danns_2d_accs = []
    train_on_taget_accs = []
    isihda_model_accs = []
    isihda_user_accs = []
    codats_accs = []
    without_adapt_accs = []
    executed_patterns = []
    num_repeats = FLAGS.num_repeats

    for pat in get_experimental_PAT():
        danns_2d_acc = 0
        train_on_taget_acc = 0
        isihda_model_acc = 0
        isihda_user_acc = 0
        codats_acc = 0
        without_adapt_acc = 0
        for _ in range(num_repeats):
            danns_2d_acc += danns_2d(pat)
            train_on_taget_acc += train_on_target(pat)
            isihda_model_acc += isih_da_model(pat)
            isihda_user_acc += isih_da_user(pat)
            codats_acc += codats(pat)
            without_adapt_acc += without_adapt(pat)
        danns_2d_accs.append(danns_2d_acc / num_repeats)
        train_on_taget_accs.append(train_on_taget_acc / num_repeats)
        isihda_model_accs.append(isihda_model_acc / num_repeats)
        isihda_user_accs.append(isihda_user_acc / num_repeats)
        codats_accs.append(codats_acc / num_repeats)
        without_adapt_accs.append(without_adapt_acc / num_repeats)
        executed_patterns.append(f"({pat.source_user},{pat.source_model})->({pat.target_user},{pat.target_model})")

    df = pd.DataFrame()
    df["PAT"] = executed_patterns
    df["Train on Target"] = train_on_taget_accs
    df["DANNs-2D"] = danns_2d_accs
    df["Isih-DA(Model => User)"] = isihda_model_accs
    df["Isih-DA(User => Model)"] = isihda_user_accs
    df["CoDATS"] = codats_accs
    df["Without Adapt"] = without_adapt_accs
    df.to_csv(f"HHAR_{str(datetime.now())}_{FLAGS.algo_name}.csv", index=False)


def get_experimental_PAT():
    import itertools
    import random

    combinations = list(itertools.product(USER_LIST, MODEL_LIST))
    valid_combinations = [
        (u1, m1, u2, m2) for (u1, m1), (u2, m2) in itertools.combinations(combinations, 2) if u1 != u2 and m1 != m2
    ]
    patterns = [
        Pattern(source_user=u1, source_model=m1, target_user=u2, target_model=m2)
        for u1, m1, u2, m2 in valid_combinations
    ]
    sampled_patterns = random.sample(patterns, 16)
    return sampled_patterns


if __name__ == "__main__":
    app.run(main)
