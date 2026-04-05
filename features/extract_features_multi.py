"""
Multi-Backbone Feature Extraction
====================================
Extracts frozen CNN/ViT features from multiple backbones for CIFAR-100,
Tiny ImageNet, and CUB-200-2011. Features are cached to disk for reuse.

Supported backbones:
  - ResNet-18 (512D)
  - ResNet-50 (2048D)
  - MobileNetV3-Small (576D)
  - EfficientNet-B0 (1280D)
  - ViT-B/16 (768D) — Vision Transformer (Dosovitskiy et al., 2020)
  - DINOv2 ViT-S/14 (384D) — Self-supervised ViT (Oquab et al., 2023)

Author: Research Study
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from typing import Tuple, Dict
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Backbone Registry ───

BACKBONES: Dict[str, Dict] = {
    "resnet18": {
        "model_fn": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "feature_dim": 512,
        "remove_head": lambda m: nn.Sequential(*list(m.children())[:-1], nn.Flatten()),
    },
    "resnet50": {
        "model_fn": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
        "feature_dim": 2048,
        "remove_head": lambda m: nn.Sequential(*list(m.children())[:-1], nn.Flatten()),
    },
    "mobilenetv3": {
        "model_fn": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "feature_dim": 576,
        "remove_head": lambda m: nn.Sequential(
            m.features,
            m.avgpool,
            nn.Flatten(),
        ),
    },
    "efficientnet": {
        "model_fn": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "feature_dim": 1280,
        "remove_head": lambda m: nn.Sequential(
            m.features,
            m.avgpool,
            nn.Flatten(),
        ),
    },
    "vit_b16": {
        "model_fn": lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
        "feature_dim": 768,
        "remove_head": "vit",  # Special handling — uses CLS token, not Sequential
    },
    "dinov2_vits14": {
        "model_fn": lambda: torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"),
        "feature_dim": 384,
        "remove_head": "dinov2",  # Special handling — already a feature extractor
    },
}


def get_device() -> torch.device:
    """Get best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_imagenet_transform(image_size: int = 224) -> transforms.Compose:
    """ImageNet normalization transform — resizes to image_size for pretrained models.

    ImageNet-pretrained backbones expect 224×224 input.  Small datasets like
    CIFAR-100 (32×32) and Tiny ImageNet (64×64) MUST be resized to 224 to
    produce high-quality features.  Without resizing, accuracy drops from
    ~64% to ~42% on CIFAR-100.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_feature_extractor(backbone_name: str, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Create a frozen feature extractor for the given backbone.

    Args:
        backbone_name: Key in BACKBONES dict
        device: torch device

    Returns:
        model: Feature extraction model (frozen, eval mode)
        feature_dim: Output dimensionality
    """
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONES.keys())}")

    config = BACKBONES[backbone_name]
    model = config["model_fn"]()
    remove_head = config["remove_head"]

    if remove_head == "vit":
        # ViT-B/16: wrap to extract CLS token from encoder output
        extractor = _ViTFeatureExtractor(model)
    elif remove_head == "dinov2":
        # DINOv2: model already outputs CLS token features directly
        extractor = model
    else:
        # CNN backbones: remove classification head
        extractor = remove_head(model)

    extractor = extractor.to(device)
    extractor.eval()

    # Freeze all parameters
    for param in extractor.parameters():
        param.requires_grad = False

    return extractor, config["feature_dim"]


class _ViTFeatureExtractor(nn.Module):
    """Wrapper for torchvision ViT to extract CLS token features."""

    def __init__(self, vit_model: nn.Module):
        super().__init__()
        self.vit = vit_model
        # Remove the classification head — we want the encoder output
        self.vit.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Extracting",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a frozen model.

    Args:
        model: Feature extraction model
        dataloader: DataLoader for the dataset
        device: torch device
        desc: Progress bar description

    Returns:
        features: (N, D) numpy array
        labels: (N,) numpy array
    """
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        features = model(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def get_or_extract_cifar100(
    backbone_name: str,
    cache_dir: str = "features/saved",
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Get (or extract and cache) features for CIFAR-100.

    Args:
        backbone_name: Backbone key
        cache_dir: Directory to cache features
        batch_size: Extraction batch size

    Returns:
        X_train, y_train, X_test, y_test, feature_dim
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{backbone_name}_cifar100.npz")

    if os.path.exists(cache_file):
        print(f"  Loading cached CIFAR-100 features: {cache_file}")
        data = np.load(cache_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"], int(data["dim"])

    print(f"  Extracting CIFAR-100 features with {backbone_name}...")
    from torchvision.datasets import CIFAR100

    device = get_device()
    model, feature_dim = create_feature_extractor(backbone_name, device)

    # Resize to 224×224 — required for ImageNet-pretrained backbones.
    # Without this, CIFAR-100 (32×32) produces ~42% accuracy instead of ~64%.
    transform = get_imagenet_transform(image_size=224)

    train_dataset = CIFAR100(root="data", train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    X_train, y_train = extract_features(model, train_loader, device, desc=f"CIFAR-100 train ({backbone_name})")
    X_test, y_test = extract_features(model, test_loader, device, desc=f"CIFAR-100 test ({backbone_name})")

    print(f"  Caching to {cache_file} (train: {X_train.shape}, test: {X_test.shape})")
    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dim=feature_dim)

    return X_train, y_train, X_test, y_test, feature_dim


def get_or_extract_tiny_imagenet(
    backbone_name: str,
    cache_dir: str = "features/tiny_imagenet",
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Get (or extract and cache) features for Tiny ImageNet.

    Args:
        backbone_name: Backbone key
        cache_dir: Directory to cache features
        batch_size: Extraction batch size

    Returns:
        X_train, y_train, X_test, y_test, feature_dim
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{backbone_name}_features.npz")

    if os.path.exists(cache_file):
        print(f"  Loading cached Tiny ImageNet features: {cache_file}")
        data = np.load(cache_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"], int(data["dim"])

    print(f"  Extracting Tiny ImageNet features with {backbone_name}...")
    from data.tiny_imagenet import TinyImageNet

    device = get_device()
    model, feature_dim = create_feature_extractor(backbone_name, device)

    # Resize to 224×224 — required for ImageNet-pretrained backbones.
    transform = get_imagenet_transform(image_size=224)

    # TinyImageNet(root=...) appends 'tiny-imagenet-200' internally, so root="data"
    data_root = "data"
    train_dataset = TinyImageNet(root=data_root, train=True, transform=transform, download=True)
    test_dataset = TinyImageNet(root=data_root, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    X_train, y_train = extract_features(model, train_loader, device, desc=f"TinyIN train ({backbone_name})")
    X_test, y_test = extract_features(model, test_loader, device, desc=f"TinyIN test ({backbone_name})")

    print(f"  Caching to {cache_file} (train: {X_train.shape}, test: {X_test.shape})")
    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dim=feature_dim)

    return X_train, y_train, X_test, y_test, feature_dim


def get_or_extract_cub200(
    backbone_name: str,
    cache_dir: str = "features/cub200",
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Get (or extract and cache) features for CUB-200-2011.

    Args:
        backbone_name: Backbone key
        cache_dir: Directory to cache features
        batch_size: Extraction batch size (smaller default — CUB has variable-size images)

    Returns:
        X_train, y_train, X_test, y_test, feature_dim
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{backbone_name}_cub200.npz")

    if os.path.exists(cache_file):
        print(f"  Loading cached CUB-200 features: {cache_file}")
        data = np.load(cache_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"], int(data["dim"])

    print(f"  Extracting CUB-200 features with {backbone_name}...")
    from data.cub200 import CUB200

    device = get_device()
    model, feature_dim = create_feature_extractor(backbone_name, device)

    # Resize to 224×224 — required for ImageNet-pretrained backbones.
    # CUB-200 images have variable sizes, so Resize(256) + CenterCrop(224) is standard.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = CUB200(root="data", train=True, transform=transform, download=True)
    test_dataset = CUB200(root="data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    X_train, y_train = extract_features(model, train_loader, device, desc=f"CUB-200 train ({backbone_name})")
    X_test, y_test = extract_features(model, test_loader, device, desc=f"CUB-200 test ({backbone_name})")

    print(f"  Caching to {cache_file} (train: {X_train.shape}, test: {X_test.shape})")
    np.savez_compressed(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dim=feature_dim)

    return X_train, y_train, X_test, y_test, feature_dim


if __name__ == "__main__":
    """Extract and cache features for all backbones × datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract and cache CNN/ViT features")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES.keys()),
                        help="Backbones to extract")
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "tiny_imagenet", "cub200"],
                        help="Datasets to extract")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    for backbone in args.backbones:
        for dataset in args.datasets:
            print(f"\n{'='*50}")
            print(f"Extracting: {backbone} × {dataset}")
            print(f"{'='*50}")
            t0 = time.perf_counter()

            if dataset == "cifar100":
                X_train, y_train, X_test, y_test, dim = get_or_extract_cifar100(
                    backbone, batch_size=args.batch_size
                )
            elif dataset == "tiny_imagenet":
                X_train, y_train, X_test, y_test, dim = get_or_extract_tiny_imagenet(
                    backbone, batch_size=args.batch_size
                )
            elif dataset == "cub200":
                X_train, y_train, X_test, y_test, dim = get_or_extract_cub200(
                    backbone, batch_size=min(args.batch_size, 64)
                )
            else:
                print(f"  Unknown dataset: {dataset}, skipping")
                continue

            elapsed = time.perf_counter() - t0
            print(f"  Done in {elapsed:.1f}s — train: {X_train.shape}, test: {X_test.shape}")
