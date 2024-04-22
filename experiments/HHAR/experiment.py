import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from ...utils import utils
from ...networks import CoDATS_F_C


if __name__ == "__main__":
    df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
    df = df[df.User=="a"]
    df = df[df.Model=="s3mini"]
    df = df[df.Device == "s3mini_2"]
    df = df.dropna(how="any")
    df = df.reset_index()
    gt_to_int_tmp = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4}
    df["gt"] = df["gt"].apply(lambda x: gt_to_int_tmp[x])
    
    X, y = utils.apply_sliding_window(df[["x", "y", "z"]].values, df["gt"].values.reshape(-1), filter_len=128, is_overlap=False)
    X = torch.tensor(X, dtype=torch.float32).to(utils.DEVICE)
    y = torch.tensor(y, dtype=torch.long).to(utils.DEVICE)
    print(X.shape)
    print(y.shape)

    codats_f_c = CoDATS_F_C(input_size=X.shape[2], output_size=len(gt_to_int_tmp))
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    out = codats_f_c(X)
    out = softmax(out)
    print(out.shape)
    loss = criterion(out, y)
    print(loss)

    loss.backward()
