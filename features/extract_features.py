"""
Feature Extraction Module

Extracts 512-dimensional features from CIFAR-100 using frozen ResNet-18.
Features are extracted ONCE and saved to disk.

Logs timing for:
- Data loading
- Model loading
- Feature extraction (train and test separately)
- Saving features
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.load_cifar100 import load_cifar100, get_dataloaders


def get_device():
    """
    Get the best available device.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class FeatureExtractor(nn.Module):
    """
    Frozen ResNet-18 feature extractor.
    Removes the final classification layer, outputs 512-dim vectors.
    """
    
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer
        # ResNet-18 architecture: conv layers -> avgpool -> fc
        # We keep everything except fc
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (batch, 3, 32, 32)
        
        Returns:
            Features of shape (batch, 512)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)
        return x


def extract_features(dataloader, model, device):
    """
    Extract features for all images in a dataloader.
    
    Args:
        dataloader: DataLoader with images
        model: Feature extraction model
        device: Device to run on
    
    Returns:
        features: numpy array of shape (N, 512)
        labels: numpy array of shape (N,)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels


def save_features(output_dir, X_train, X_test, y_train, y_test, timings=None):
    """
    Save extracted features and timing log to disk.
    
    Args:
        output_dir: Directory to save features
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        timings: Optional dict with timing information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save timing log
    if timings is not None:
        with open(os.path.join(output_dir, 'extraction_timing.json'), 'w') as f:
            json.dump(timings, f, indent=2)
    
    print(f"Features saved to {output_dir}")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")


def load_features(feature_dir):
    """
    Load previously extracted features from disk.
    
    Args:
        feature_dir: Directory containing saved features
    
    Returns:
        X_train, X_test, y_train, y_test: numpy arrays
    """
    X_train = np.load(os.path.join(feature_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(feature_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(feature_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(feature_dir, 'y_test.npy'))
    
    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to extract and save features with timing.
    """
    # Configuration
    data_root = './data'
    feature_dir = './features/saved'
    batch_size = 128
    num_workers = 4
    
    # Timing dictionary
    timings = {}
    total_start = time.perf_counter()
    
    # Check if features already exist
    if os.path.exists(os.path.join(feature_dir, 'X_train.npy')):
        print("Features already exist. To re-extract, delete the saved directory.")
        return
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-100...")
    t_start = time.perf_counter()
    train_dataset, test_dataset = load_cifar100(root=data_root)
    train_loader, test_loader = get_dataloaders(
        train_dataset, test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    timings['data_loading_sec'] = time.perf_counter() - t_start
    print(f"  Data loading time: {timings['data_loading_sec']:.2f}s")
    
    # Create model
    print("Loading ResNet-18 (frozen, ImageNet pretrained)...")
    t_start = time.perf_counter()
    model = FeatureExtractor().to(device)
    timings['model_loading_sec'] = time.perf_counter() - t_start
    print(f"  Model loading time: {timings['model_loading_sec']:.2f}s")
    
    # Extract training features
    print("\nExtracting training features...")
    t_start = time.perf_counter()
    X_train, y_train = extract_features(train_loader, model, device)
    timings['train_extraction_sec'] = time.perf_counter() - t_start
    print(f"  Training extraction time: {timings['train_extraction_sec']:.2f}s")
    
    # Extract test features
    print("\nExtracting test features...")
    t_start = time.perf_counter()
    X_test, y_test = extract_features(test_loader, model, device)
    timings['test_extraction_sec'] = time.perf_counter() - t_start
    print(f"  Test extraction time: {timings['test_extraction_sec']:.2f}s")
    
    # Verify shapes
    assert X_train.shape == (50000, 512), f"Unexpected train shape: {X_train.shape}"
    assert X_test.shape == (10000, 512), f"Unexpected test shape: {X_test.shape}"
    assert y_train.shape == (50000,), f"Unexpected train label shape: {y_train.shape}"
    assert y_test.shape == (10000,), f"Unexpected test label shape: {y_test.shape}"
    
    # Verify label range
    assert y_train.min() == 0 and y_train.max() == 99, "Invalid training labels"
    assert y_test.min() == 0 and y_test.max() == 99, "Invalid test labels"
    
    # Save features
    print("\nSaving features...")
    t_start = time.perf_counter()
    save_features(feature_dir, X_train, X_test, y_train, y_test, timings)
    timings['saving_sec'] = time.perf_counter() - t_start
    print(f"  Saving time: {timings['saving_sec']:.2f}s")
    
    # Total time
    timings['total_sec'] = time.perf_counter() - total_start
    
    # Update timing file with total
    with open(os.path.join(feature_dir, 'extraction_timing.json'), 'w') as f:
        json.dump(timings, f, indent=2)
    
    print("\n" + "="*50)
    print("Feature extraction complete!")
    print("="*50)
    print(f"Total time: {timings['total_sec']:.2f}s")
    print(f"  Data loading:    {timings['data_loading_sec']:.2f}s")
    print(f"  Model loading:   {timings['model_loading_sec']:.2f}s")
    print(f"  Train extraction: {timings['train_extraction_sec']:.2f}s")
    print(f"  Test extraction:  {timings['test_extraction_sec']:.2f}s")
    print(f"  Saving:          {timings['saving_sec']:.2f}s")


if __name__ == "__main__":
    main()
