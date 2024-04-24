import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from ...utils import utils
from ...networks import CoDATS_F_C
GT_TO_INT = {"bike": 0, "stairsup": 1, "stairsdown": 2, "stand": 3, "walk": 4, "sit": 5}
USER_LIST = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
MODEL_LIST = ["nexus4", "s3", "samsungold", "s3mini"]

def get_data_for_uda(user):
    assert model in MODEL_LIST
    assert user in USER_LIST

    accelerometer_df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_accelerometer.csv")
    accelerometer_df = accelerometer_df.add_suffix('_accele')
    gyroscope_df = pd.read_csv("./domain-invariant-learning/experiments/HHAR/data/heterogeneity+activity+recognition/Activity recognition exp/Activity recognition exp/Phones_gyroscope.csv")
    gyroscope_df = gyroscope_df.add_suffix('_gyro')
    df = pd.merge(accelerometer_df, gyroscope_df, left_on=["Arrival_Time_accele", "User_accele", "Device_accele"], right_on=["Arrival_Time_gyro", "User_gyro", "Device_gyro"],how="left")
    df[["x_gyro", "y_gyro", "z_gyro"]] = df[["x_gyro", "y_gyro", "z_gyro"]].interpolate()
    df = df[df.User_accele==user]
    df = df.dropna(how="any")
    df = df.reset_index()
    df["gt_accele"] = df["gt_accele"].apply(lambda x: GT_TO_INT[x])
    X, y = utils.apply_sliding_window(df[["x_accele", "y_accele", "z_accele", "x_gyro", "y_gyro", "z_gyro"]].values, df["gt_accele"].values.reshape(-1), filter_len=128, is_overlap=False)
    return X, y

if __name__ == "__main__":
    # Load Data
    X, y = get_data_for_uda(user="a")

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
    codats_f_c = CoDATS_F_C(input_size=X.shape[2], output_size=6)
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    optimizer = optim.Adam(codats_f_c.parameters(), lr=0.0001)
    # Epoch Training
    losses = []
    from tqdm import tqdm
    for _ in tqdm(range(500)):
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

        plt.figure()
        plt.plot(test_X[i, :, 3].cpu().detach().numpy(), label="gyro_x")
        plt.plot(test_X[i, :, 4].cpu().detach().numpy(), label="gyro_y")
        plt.plot(test_X[i, :, 5].cpu().detach().numpy(), label="gyro_z")
        plt.title(f"Pred: {out.argmax(dim=1)[i].cpu().numpy()} vs GT: {test_y[i].cpu().numpy()}")
        plt.show()