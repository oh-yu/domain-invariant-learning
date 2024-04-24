import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import utils
from ...networks import IsihDanns


GT_TO_INT = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4, "sit": 5}
USER_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
MODEL_LIST = ["nexus4", "s3", "samsungold", "s3mini"]

def get_data_for_uda(user, model, is_targer_prime: bool = False):
    assert model in MODEL_LIST
    assert user in USER_LIST

    df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
    df = df[df.User==user]
    # df = df[df.Model==model]
    df = df.dropna(how="any")
    df = df.reset_index()
    df["gt"] = df["gt"].apply(lambda x: GT_TO_INT[x])
    scaler = StandardScaler()
    if not is_targer_prime:
        df[["x", "y", "z"]] = scaler.fit_transform(df[["x", "y", "z"]])
    X, y = utils.apply_sliding_window(df[["x", "y", "z"]].values, df["gt"].values.reshape(-1), filter_len=128, is_overlap=False)
    return X, y

if __name__ == "__main__":
    # Load Data
    source_X, source_y_task = get_data_for_uda(user="b", model="nexus4")
    target_X, target_y_task = get_data_for_uda(user="d", model="nexus4")
    target_prime_X, target_prime_y_task = get_data_for_uda(user="b", model="s3")
    

    # Algo1: Inter-devices DA
    source_loader, target_loader, _, _, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=128, shuffle=True
    )
    isih_dann = IsihDanns(
        input_size = source_X.shape[2],
        hidden_size=128,
        lr_dim1 = 0.0001,
        lr_dim2=0.00005,
        num_epochs_dim1=3000,
        num_epochs_dim2=10,
        output_size=len(GT_TO_INT)
    )
    isih_dann.fit_1st_dim(source_loader, target_loader, target_X, target_y_task)
    # TODO: multi-class


    # Algo2: Inter-users DA
    # TODO: Scaler
    

    # Algo3: Evaluation

    

