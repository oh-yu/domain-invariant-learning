import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from ...networks import IsihDanns

from ...utils import utils


class Reshape(object):
    def __call__(self, img):
        padding = torch.zeros(3, 32, 32)
        padding[:, 2:30, 2:30] = img.repeat(3, 1, 1)
        return padding
    # TODO: Understand this style implementation


class CustomUDADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, source_or_target):
        self.dataset = dataset
        self.source_or_target = source_or_target
        if source_or_target == "source":
            self.domain_label = 0
        elif source_or_target == "target":
            self.domain_label = 1
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        domain_label = self.domain_label

        if self.source_or_target == "source":
            return image, torch.tensor([label, float(domain_label)])
        elif self.source_or_target == "target":
            return image, torch.tensor(domain_label, dtype=torch.float32)

    # TODO: Understand this style implementation


def get_image_data_for_uda(name="MNIST", source_or_target="source"):
    assert name in ["MNIST", "MNIST-M", "SVHN"]

    if name == "MNIST":
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            Reshape(),
        ])
        # TODO: Understand transforms.Compose
        train_data = datasets.MNIST(root="./domain-invariant-learning/experiments/MNIST/data/MNIST", train=True, download=True, transform=custom_transform)
        train_data = CustomUDADataset(train_data, source_or_target)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
        return train_loader
    
    elif name == "MNIST-M":
        transform = transforms.ToTensor()
        train_data = ImageFolder(root='./domain-invariant-learning/experiments/MNIST/data/MNIST-M/training', transform=transform)
        train_data = CustomUDADataset(train_data, source_or_target)

        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        return train_loader, train_data

    elif name == "SVHN":
        transform = transforms.ToTensor()
        train_data = torchvision.datasets.SVHN(
            './data/SVHN', 
            split='train',
            download=True,
            transform=transform)
        train_data = CustomUDADataset(train_data, source_or_target)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        test_data = torchvision.datasets.SVHN(
            "./data/SVHN",
            split="test",
            download=True,
            transform=transform)
        test_data = CustomUDADataset(test_data, source_or_target)
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        return train_loader, test_loader


if __name__ == "__main__":
    # Load Data
    source_loader = get_image_data_for_uda("MNIST", "source")
    target_loader, target_data = get_image_data_for_uda("MNIST-M", "target")
    target_loader_gt, _ = get_image_data_for_uda("MNIST-M", "source")
    target_prime_loader = get_image_data_for_uda("SVHN")

    # Model Init
    isih_dann = IsihDanns(
        input_size=None,
        hidden_size=None,
        lr_dim1=0.0001,
        lr_dim2=0.00005,
        num_epochs_dim1=50,
        num_epochs_dim2=1,
        experiment="MNIST"
    )

    # Algo1 inter-colors DA
    target_X = torch.cat([X for X, _ in target_loader], dim=0)
    target_y_task = torch.cat([y[:, 0] for _, y in target_loader_gt], dim=0)
    target_X = torch.tensor(target_X, dtype=torch.float32).to(utils.DEVICE)
    target_y_task = torch.tensor(target_y_task, dtype=torch.long).to(utils.DEVICE)
    isih_dann.fit_1st_dim(source_loader, target_loader, target_X, target_y_task)
    
    # Algo2 inter-reals DA


    # Algo3 Evaluation