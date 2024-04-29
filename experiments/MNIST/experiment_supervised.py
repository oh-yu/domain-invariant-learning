import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from ...networks import Conv2d, DomainDecoder


class Reshape(object):
    def __call__(self, img):
        padding = torch.zeros(3, 32, 32)
        padding[:, 2:30, 2:30] = img.repeat(3, 1, 1)
        return padding
    # TODO: Understand this style implementation

def get_image_data_for_uda(name="MNIST"):
    if name == "MNIST":
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            Reshape(),
        ])
        # TODO: Understand transforms.Compose
        train_data = datasets.MNIST(root="./data/MNIST", train=True, download=True, transform=custom_transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        return train_loader

    elif name == "SVHN":
        transform = transforms.ToTensor()
        train_data = torchvision.datasets.SVHN(
            './data/SVHN', 
            split='train',
            download=True,
            transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        test_data = torchvision.datasets.SVHN(
            "./data/SVHN",
            split="test",
            download=True,
            transform=transform)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        return train_loader, test_loader





if __name__ == "__main__":
    # Load Data
    train_loader = get_image_data_for_uda("MNIST")

    # Model Init
    feature_extractor = Conv2d()
    task_classifier = DomainDecoder(input_size=1600, output_size=10, fc2_size=50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(task_classifier.parameters()), lr=1e-3)

    # Training
    for _ in range(1):
        for X_batch, y_batch in train_loader:
            out = task_classifier.predict_proba(feature_extractor(X_batch))
            loss = criterion(out, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if _ % 10 == 0:
            print(f"Epoch {_}: Loss {loss}")

    # Evaluation
    # for X_batch, y_batch in test_loader:
    #     out = task_classifier.predict(feature_extractor(X_batch))
    #     acc = sum(out == y_batch) / len(y_batch)
    #     print(acc)
    #     break