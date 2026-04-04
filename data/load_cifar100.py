"""
CIFAR-100 Data Loading Module

Loads CIFAR-100 dataset using torchvision with standard train/test split.
Does NOT modify labels or mix train/test data.
"""

import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader


def get_transforms(image_size: int = 224):
    """
    Returns transforms for feature extraction with ImageNet-pretrained models.

    ImageNet-pretrained backbones expect 224×224 input.  CIFAR-100 images
    (32×32) MUST be resized to produce high-quality features.

    Args:
        image_size: Target spatial size.  Default 224 (standard for ImageNet
            pretrained models).
    """
    steps = []
    steps.append(transforms.Resize(image_size))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    ))
    return transforms.Compose(steps)


def load_cifar100(root='./data', download=True, image_size: int = 224):
    """
    Load CIFAR-100 train and test datasets.
    
    Args:
        root: Directory to store/load data
        download: Whether to download if not present
        image_size: Target spatial size for transforms (default 224)
    
    Returns:
        train_dataset: Training set (50,000 images)
        test_dataset: Test set (10,000 images)
    """
    transform = get_transforms(image_size=image_size)
    
    train_dataset = CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = CIFAR100(
        root=root,
        train=False,
        download=download,
        transform=transform
    )
    
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size=128, num_workers=4):
    """
    Create DataLoaders for batch processing during feature extraction.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size for loading
        num_workers: Number of worker processes
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for reproducibility
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test
    train_ds, test_ds = load_cifar100()
    print(f"Training samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Number of classes: {len(train_ds.classes)}")
    print(f"Image shape: {train_ds[0][0].shape}")
