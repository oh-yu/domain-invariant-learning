import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from ...networks import Conv2d, DomainDecoder


class Reshape(object):
    def __call__(self, img):
        padding = torch.zeros(3, 32, 32)
        padding[:, 2:30, 2:30] = img.repeat(3, 1, 1)
        return padding
    # TODO: Understand this style implementation


class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
    def __len__(self):
        return len(self.mnist_dataset)
    def __getitem__(self, idx):
        image, label = self.mnist_dataset[idx]
        domain_label = 0
        return image, (label, domain_label)


def get_image_data_for_uda(name="MNIST"):
    assert name in ["MNIST", "MNIST-M", "SVHN"]

    if name == "MNIST":
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            Reshape(),
        ])
        # TODO: Understand transforms.Compose
        train_data = datasets.MNIST(root="./domain-invariant-learning/experiments/MNIST/data/MNIST", train=True, download=True, transform=custom_transform)
        train_data = CustomMNISTDataset(train_data)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
        return train_loader
    
    elif name == "MNIST-M":
        transform = transforms.ToTensor()
        train_data = ImageFolder(root='./domain-invariant-learning/experiments/MNIST/data/MNIST-M/training', transform=transform)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
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
    source_loader = get_image_data_for_uda("MNIST")
    target_loader = get_image_data_for_uda("MNIST-M")
    target_prime_loader = get_image_data_for_uda("SVHN")

    for X, y in source_loader:
        print(y[0])
        print(y[1])
        break

    # Algo1 inter-colors DA


    # Algo2 inter-real DA


    # Algo3 Evaluation