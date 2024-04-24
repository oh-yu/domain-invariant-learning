import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import utils
from ...networks import IsihDanns


def get_data_for_uda(user, model):
    assert model in ["nexus4", "s3", "samsungold", "s3mini"]
    assert user in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
    df = df[df.User==user]
    df = df[df.Model==model]
    df = df.dropna(how="any")
    df = df.reset_index()
    gt_to_int_tmp = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4}
    df["gt"] = df["gt"].apply(lambda x: gt_to_int_tmp[x])
    X, y = utils.apply_sliding_window(df[["x", "y", "z"]].values, df["gt"].values.reshape(-1), filter_len=128, is_overlap=False)
    

if __name__ == "__main__":
    # Load Data
    source_X, source_y_task = get_data_for_uda(user="a", model="nexus4")
    target_X, target_y_task = get_data_for_uda(user="a", model="s3")
    target_prime_X, target_prime_y_task = get_data_for_uda(user="b", model="s3")
    scaler = StandardScaler()
    source_X = scaler.fit_transform(source_X)
    target_X = scaler.fit_transform(target_X)
    # Algo1: Inter-devices DA
    source_loader, target_loader, _, _, target_X, target_y_task = utils.get_loader(
        source_X, target_X, source_y_task, target_y_task, batch_size=4, shuffle=True
    )
    isih_dann = IsihDanns(
        input_size = source_X.shape[2],
        hidden_size=128,
        lr_dim1 = 0.0001,
        lr_dim2=0.00005,
        num_epochs_dim1=200,
        num_epochs_dim2=100
    )
    isih_dann.fit_1st_dim(source_loader, target_loader, target_X, target_y_task)
    # TODO: multi-class


    # Algo2: Inter-users DA
    # TODO: Scaler
    

    # Algo3: Evaluation

    
