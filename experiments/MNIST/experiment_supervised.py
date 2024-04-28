from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

if __name__ == "__main__":
    # Load Data
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
    # Model Init

    # Training

    # Evaluation