import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from ...utils import utils
from ...networks import IsihDanns


GT_TO_INT = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4, "sit": 5}
USER_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
MODEL_LIST = ["nexus4", "s3", "samsungold", "s3mini"]


def get_data_for_uda(user, model, is_targer_prime: bool = False):
    assert model in MODEL_LIST
    assert user in USER_LIST

    accelerometer_df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
    accelerometer_df = accelerometer_df.add_suffix('_accele')
    gyroscope_df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv")
    gyroscope_df = gyroscope_df.add_suffix('_gyro')
    df = pd.merge(accelerometer_df, gyroscope_df, left_on=["Arrival_Time_accele", "User_accele", "Device_accele"], right_on=["Arrival_Time_gyro", "User_gyro", "Device_gyro"],how="left")
    df[["x_gyro", "y_gyro", "z_gyro"]] = df[["x_gyro", "y_gyro", "z_gyro"]].interpolate()
    df[["x_accele", "y_accele", "z_accele"]] = df[["x_accele", "y_accele", "z_accele"]].interpolate()
    df = df[df.User_accele==user]
    # df = df[df.Model_accele==model]
    df = df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro", "gt_accele"]].dropna(how="any")
    df = df.reset_index()
    df["gt_accele"] = df["gt_accele"].apply(lambda x: GT_TO_INT[x])
    scaler = StandardScaler()
    if not is_targer_prime:
        df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]] = scaler.fit_transform(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]])
        X, y = utils.apply_sliding_window(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values, df["gt_accele"].values.reshape(-1), filter_len=128, is_overlap=False)
        return X, y
    else:
        train_X, test_X, train_y, test_y = train_test_split(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values, df["gt_accele"].values, test_size=0.2, shuffle=False)
        # Shuffle = False, since data has meaning of time series
    
        train_X, test_X = scaler.fit_transform(train_X), scaler.transform(test_X)
        train_X, train_y = utils.apply_sliding_window(train_X, train_y.reshape(-1), filter_len=128, is_overlap=False)
        test_X, test_y = utils.apply_sliding_window(test_X, test_y.reshape(-1), filter_len=128, is_overlap=False)
        return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    # Load Data
    source_X, source_y_task = get_data_for_uda(user="c", model="nexus4")
    target_X, target_y_task = get_data_for_uda(user="d", model="nexus4")
    train_target_prime_X, train_target_prime_y_task, test_target_prime_X, test_target_prime_y_task = get_data_for_uda(user="d", model="s3", is_targer_prime=True)
    
    # Algo1: Inter-devices DA
    source_loader, target_loader, _, _, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True
    )
    isih_dann = IsihDanns(
        input_size = source_X.shape[2],
        hidden_size=128,
        lr_dim1 = 0.0001,
        lr_dim2=0.00005,
        num_epochs_dim1=30,
        num_epochs_dim2=30,
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
    print(f"Accuracy: {acc}")
    

