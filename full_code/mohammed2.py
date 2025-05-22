from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.transforms import Compose

from .config import BATCH_SIZE, DATA_DIR


def loading(
    train_transform: Compose, val_transform: Compose
) -> tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR, transform=train_transform
    )  # loads the dataset from a PetImages folder which has two folders Cat,Dog

    train_size = int(0.8 * len(full_dataset))  # training data set size = 80%
    val_size = len(full_dataset) - train_size  # validation data set size = 20%
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # it still references the original full_dataset, which was using train_transform.
    # This line overrides the transform on val_dataset to use val_transform instead.
    # Prevents validation images from being distorted by training-time randomness.
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    return train_loader, val_loader, full_dataset
