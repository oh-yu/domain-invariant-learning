from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from ...networks import Conv2d, DomainDecoder

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
    feature_extractor = Conv2d()
    task_classifier = DomainDecoder(input_size=100, output_size=10, fc2_size=50)
    # TODO: check input_size
    # TODO: init criterion, optimizer

    # Training
    for X_batch, y_batch in train_loader:
        out = feature_extractor(X_batch)
        print(out.shape)



    # Evaluation