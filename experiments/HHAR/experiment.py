import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from ...utils import utils
from ...networks import Codats, IsihDanns, CoDATS_F_C


GT_TO_INT = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4, "sit": 5}
USER_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
MODEL_LIST = ["nexus4", "s3", "samsungold", "s3mini"]
ACCELEROMETER_DF = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
ACCELEROMETER_DF = ACCELEROMETER_DF.add_suffix('_accele')
GYROSCOPE_DF = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv")
GYROSCOPE_DF = GYROSCOPE_DF.add_suffix('_gyro')


def get_data_for_uda(user, model, is_targer_prime: bool = False):
    assert model in MODEL_LIST
    assert user in USER_LIST
    df = pd.merge(ACCELEROMETER_DF, GYROSCOPE_DF, left_on=["Arrival_Time_accele", "User_accele", "Device_accele"], right_on=["Arrival_Time_gyro", "User_gyro", "Device_gyro"],how="left")
    df[["x_gyro", "y_gyro", "z_gyro"]] = df[["x_gyro", "y_gyro", "z_gyro"]].interpolate()
    df[["x_accele", "y_accele", "z_accele"]] = df[["x_accele", "y_accele", "z_accele"]].interpolate()
    df = df[df.User_accele==user]
    df = df[df.Model_accele==model]
    df = df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro", "gt_accele"]].dropna(how="any")
    df = df.reset_index()
    df["gt_accele"] = df["gt_accele"].apply(lambda x: GT_TO_INT[x])
    scaler = StandardScaler()
    if not is_targer_prime:
        df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]] = scaler.fit_transform(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]])
        X, y = utils.apply_sliding_window(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values, df["gt_accele"].values.reshape(-1), filter_len=128, is_overlap=False)
        return X, y
    else:
        X, y = utils.apply_sliding_window(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values, df["gt_accele"].values.reshape(-1), filter_len=128, is_overlap=False)
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, stratify=y)
        # Shuffle = False, since data has meaning of time series
        train_N, T, H = train_X.shape
        test_N = test_X.shape[0]
        train_X, test_X = train_X.reshape(train_N*T, H), test_X.reshape(test_N*T, H)
        train_X, test_X = scaler.fit_transform(train_X), scaler.transform(test_X)
        train_X, test_X = train_X.reshape(train_N, T, H), test_X.reshape(test_N, T, H)
        return train_X, train_y, test_X, test_y


def isih_da_user(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    target_X, target_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.target_model, is_targer_prime=True)

    # Algo1: Inter-user DA
    source_loader, target_loader, _, _, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True
    )
    isih_dann = IsihDanns(
        input_size = source_X.shape[2],
        hidden_size=128,
        lr_dim1 = 0.0001,
        lr_dim2=0.0001,
        num_epochs_dim1=150,
        num_epochs_dim2=50,
        output_size=len(GT_TO_INT)
    )
    isih_dann.fit_1st_dim(source_loader, target_loader, target_X, target_y_task)
    pred_y_task = isih_dann.predict_proba(target_X, is_1st_dim=True)

    # Algo2: Inter-models DA
    source_X = target_X.cpu().detach().numpy()
    source_y_task = pred_y_task.cpu().detach().numpy()
    source_loader, target_loader, _, _, _, _ = utils.get_loader(
        source_X, train_target_prime_X, source_y_task, train_target_prime_y_task, batch_size=128, shuffle=True
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    isih_dann.fit_2nd_dim(source_loader, target_loader, test_target_prime_X, test_target_prime_y_task)
    isih_dann.set_eval()
    pred_y_task = isih_dann.predict(test_target_prime_X, is_1st_dim=False)

    # Algo3: Evaluation
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task) 
    return acc


def isih_da_model(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    target_X, target_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.target_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.target_model, is_targer_prime=True)

    # Algo1: Inter-models DA
    source_loader, target_loader, _, _, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True
    )
    isih_dann = IsihDanns(
        input_size = source_X.shape[2],
        hidden_size=128,
        lr_dim1 = 0.0001,
        lr_dim2=0.0001,
        num_epochs_dim1=150,
        num_epochs_dim2=50,
        output_size=len(GT_TO_INT)
    )
    isih_dann.fit_1st_dim(source_loader, target_loader, target_X, target_y_task)
    pred_y_task = isih_dann.predict_proba(target_X, is_1st_dim=True)

    # Algo2: Inter-users DA
    source_X = target_X.cpu().detach().numpy()
    source_y_task = pred_y_task.cpu().detach().numpy()
    source_loader, target_loader, _, _, _, _ = utils.get_loader(
        source_X, train_target_prime_X, source_y_task, train_target_prime_y_task, batch_size=128, shuffle=True
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    isih_dann.fit_2nd_dim(source_loader, target_loader, test_target_prime_X, test_target_prime_y_task)
    isih_dann.set_eval()
    pred_y_task = isih_dann.predict(test_target_prime_X, is_1st_dim=False)

    # Algo3: Evaluation
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task) 
    return acc


def codats(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.target_model, is_targer_prime=True)

    # Direct Inter-Users and Inter-models DA
    source_loader, target_loader, _, _, train_target_prime_X, train_target_prime_y_task = utils.get_loader(
        source_X, train_target_prime_X, source_y_task, train_target_prime_y_task, batch_size=128, shuffle=True
    )
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.float32)
    test_target_prime_X = test_target_prime_X.to(utils.DEVICE)
    test_target_prime_y_task = test_target_prime_y_task.to(utils.DEVICE)

    codats = Codats(input_size=source_X.shape[2], hidden_size=128, lr=0.0001, num_epochs=200, num_classes=len(GT_TO_INT))
    codats.fit(source_loader, target_loader, test_target_prime_X, test_target_prime_y_task)
    codats.set_eval()
    pred_y_task = codats.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc


def without_adapt(pattern):
    # Load Data
    source_X, source_y_task = get_data_for_uda(user=pattern.source_user, model=pattern.source_model)
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.target_model, is_targer_prime=True)

    # Without Adapt
    source_loader, _, _, _, _, _ = utils.get_loader(
        source_X, train_target_prime_X, source_y_task, train_target_prime_y_task, batch_size=128, shuffle=True
    )  
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32)
    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.float32)
    test_target_prime_X = test_target_prime_X.to(utils.DEVICE)
    test_target_prime_y_task = test_target_prime_y_task.to(utils.DEVICE)

    without_adapt = CoDATS_F_C(input_size=source_X.shape[2], output_size=len(GT_TO_INT))
    without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    without_adapt = utils.fit_without_adaptation(
        source_loader=source_loader,
        task_classifier=without_adapt,
        task_optimizer=without_adapt_optimizer,
        criterion=criterion,
        num_epochs=200,
    )
    pred_y_task = without_adapt.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def train_on_target(pattern):
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(user=pattern.target_user, model=pattern.target_model, is_targer_prime=True)

    train_target_prime_X = torch.tensor(train_target_prime_X, dtype=torch.float32).to(utils.DEVICE)
    train_target_prime_y_task = torch.tensor(train_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    test_target_prime_X = torch.tensor(test_target_prime_X, dtype=torch.float32).to(utils.DEVICE)

    test_target_prime_y_task = torch.tensor(test_target_prime_y_task, dtype=torch.long).to(utils.DEVICE)
    target_prime_ds = TensorDataset(train_target_prime_X, train_target_prime_y_task)
    target_prime_loader = DataLoader(target_prime_ds, batch_size=128, shuffle=True)

    train_on_target = CoDATS_F_C(input_size=train_target_prime_X.shape[2], output_size=len(GT_TO_INT))
    train_on_target_optimizer = optim.Adam(train_on_target.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(200):
        for target_prime_X_batch, target_prime_y_task_batch in target_prime_loader:
            pred_y_task = train_on_target.predict_proba(target_prime_X_batch)

            loss = criterion(pred_y_task, target_prime_y_task_batch)

            train_on_target_optimizer.zero_grad()
            loss.backward()
            train_on_target_optimizer.step()
    train_on_target.eval()
    
    pred_y_task = train_on_target.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()

if __name__ == "__main__":
    # TODO: Remove
    class Pattern:
        def __init__(self):
            self.source_user = "c"
            self.source_model = "nexus4"
            self.target_user = "d"
            self.target_model = "nexus4"
    pat = Pattern()

    print(train_on_target(pat))
    print(isih_da_model(pat))
    print(isih_da_user(pat))
    print(codats(pat))
    print(without_adapt(pat))
    