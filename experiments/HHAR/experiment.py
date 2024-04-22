import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import utils
from ...networks import CoDATS_F_C


if __name__ == "__main__":
    # Load Data
    df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
    df = df[df.User=="a"]
    df = df[df.Model=="s3mini"]
    df = df[df.Device == "s3mini_2"]
    df = df.dropna(how="any")
    df = df.reset_index()
    gt_to_int_tmp = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4}
    df["gt"] = df["gt"].apply(lambda x: gt_to_int_tmp[x]) 
    X, y = utils.apply_sliding_window(df[["x", "y", "z"]].values, df["gt"].values.reshape(-1), filter_len=128, is_overlap=False)

    # train_test_split
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
    train_X = torch.tensor(train_X, dtype=torch.float32).to(utils.DEVICE)
    train_y = torch.tensor(train_y, dtype=torch.long).to(utils.DEVICE)
    test_X = torch.tensor(test_X, dtype=torch.float32).to(utils.DEVICE)
    test_y = torch.tensor(test_y, dtype=torch.long).to(utils.DEVICE)
    # Data Loader
    ds = TensorDataset(train_X, train_y)
    data_loader = DataLoader(ds, batch_size=4, shuffle=True)

    # Model Init   
    codats_f_c = CoDATS_F_C(input_size=X.shape[2], output_size=len(gt_to_int_tmp))
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    optimizer = optim.Adam(codats_f_c.parameters(), lr=0.0001)
    # Epoch Training
    losses = []
    for _ in range(500):
        for X, y in data_loader:
            out = codats_f_c(X)
            out = softmax(out)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    plt.plot(losses)
    plt.show()
    # Evaluation
    with torch.no_grad():
        out = codats_f_c(test_X)
        out = softmax(out)
        accuracy = sum(out.argmax(dim=1) == test_y) / len(test_y)
        print(f"Accuracy: {accuracy}")
    
    # Visualization
    for i in range(test_X.shape[0]):
        plt.plot(test_X[i, :, 0].cpu().detach().numpy(), label="x")
        plt.plot(test_X[i, :, 1].cpu().detach().numpy(), label="y")
        plt.plot(test_X[i, :, 2].cpu().detach().numpy(), label="z")
        plt.title(f"Pred: {out.argmax(dim=1)[i].cpu().numpy()} vs GT: {test_y[i].cpu().numpy()}")
        plt.show()
