import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from ...networks import IsihDanns, Dann, Dann_F_C

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


def get_image_data_for_uda(name="MNIST"):
    assert name in ["MNIST", "MNIST-M", "SVHN"]

    if name == "MNIST":
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
            Reshape(),
        ])
        # TODO: Understand transforms.Compose
        train_data = datasets.MNIST(root="./domain-invariant-learning/experiments/MNIST/data/MNIST", train=True, download=True, transform=custom_transform)
        train_data = CustomUDADataset(train_data, "source")
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
        return train_loader
    
    elif name == "MNIST-M":
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        imagefolder_data = ImageFolder(root='./domain-invariant-learning/experiments/MNIST/data/MNIST-M/training', transform=custom_transform)
        train_data = CustomUDADataset(imagefolder_data, "target")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

        train_data_gt = CustomUDADataset(imagefolder_data, "source")
        train_loader_gt = DataLoader(train_data_gt, batch_size=128, shuffle=False)
        return train_loader, train_loader_gt

    elif name == "SVHN":
        custom_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = torchvision.datasets.SVHN(
            './data/SVHN', 
            split='train',
            download=True,
            transform=custom_transform)
        train_data = CustomUDADataset(train_data, "target")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_data = torchvision.datasets.SVHN(
            "./data/SVHN",
            split="test",
            download=True,
            transform=custom_transform)
        test_data = CustomUDADataset(test_data, "source")
        test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
        return train_loader, test_loader


def isih_da_real():
    # Load Data
    source_loader = get_image_data_for_uda("MNIST")
    target_loader, target_loader_gt = get_image_data_for_uda("MNIST-M")
    train_target_prime_loader, test_target_prime_loader_gt = get_image_data_for_uda("SVHN")

    # Model Init
    isih_dann = IsihDanns(
        input_size=None,
        hidden_size=None,
        lr_dim1=0.0001,
        lr_dim2=0.0005,
        num_epochs_dim1=100,
        num_epochs_dim2=100,
        experiment="MNIST",
        is_target_weights=False
    )

    # Algo1 inter-colors DA
    target_X = torch.cat([X for X, _ in target_loader_gt], dim=0)
    target_y_task = torch.cat([y[:, 0] for _, y in target_loader_gt], dim=0)
    target_X = torch.tensor(target_X, dtype=torch.float32)
    target_y_task = torch.tensor(target_y_task, dtype=torch.long)
    isih_dann.fit_1st_dim(source_loader, target_loader, target_X, target_y_task)
    pred_y_task = isih_dann.predict_proba(target_X, is_1st_dim=True)

    # Algo2 inter-reals DA
    domain_labels = torch.ones(pred_y_task.shape[0]).reshape(-1, 1)
    pred_y_task = torch.cat((pred_y_task, domain_labels), dim=1)
    source_ds = TensorDataset(target_X, pred_y_task)
    source_loader = DataLoader(source_ds, batch_size=128, shuffle=True)
    test_target_prime_X = torch.cat([X for X, _ in test_target_prime_loader_gt], dim=0)
    test_target_prime_y_task = torch.cat([y[:, 0] for _, y in test_target_prime_loader_gt], dim=0)
    isih_dann.fit_2nd_dim(source_loader, train_target_prime_loader, test_target_prime_X, test_target_prime_y_task)
    
    # Algo3 Eval
    isih_dann.set_eval()
    pred_y_task = isih_dann.predict(test_target_prime_X, is_1st_dim=False)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def dann():
    # Load Data
    source_loader = get_image_data_for_uda("MNIST")
    train_target_prime_loader, test_target_prime_loader_gt = get_image_data_for_uda("SVHN")

    # Model Init
    dann = Dann(
        task_fc1_size=3072,
        task_fc2_size=2048,
        domain_fc1_size=1024,
        domain_fc2_size=1024,
        output_size=10,
        input_size=1152,
        num_epochs=10,
    )
    # Fit DANN
    test_target_prime_X = torch.cat([X for X, _ in test_target_prime_loader_gt], dim=0)
    test_target_prime_y_task = torch.cat([y[:, 0] for _, y in test_target_prime_loader_gt], dim=0)
    dann.fit(
        source_loader,
        train_target_prime_loader,
        test_target_prime_X,
        test_target_prime_y_task,
    )

    # Eval
    dann.set_eval()
    pred_y_task = dann.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


def without_adapt():
    # Load Data
    source_loader = get_image_data_for_uda("MNIST")
    train_target_prime_loader, test_target_prime_loader_gt = get_image_data_for_uda("SVHN")

    # Model Init
    without_adapt = Dann_F_C()
    without_adapt_optimizer = optim.Adam(without_adapt.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Fit
    without_adapt = utils.fit_without_adaptation(source_loader, without_adapt, without_adapt_optimizer, criterion, output_size=10, num_epochs=10)
    # Eval
    test_target_prime_X = torch.cat([X for X, _ in test_target_prime_loader_gt], dim=0)
    test_target_prime_y_task = torch.cat([y[:, 0] for _, y in test_target_prime_loader_gt], dim=0)
    pred_y_task = without_adapt.predict(test_target_prime_X)
    acc = sum(pred_y_task == test_target_prime_y_task) / len(test_target_prime_y_task)
    return acc.item()


if __name__ == "__main__":
    dann()
