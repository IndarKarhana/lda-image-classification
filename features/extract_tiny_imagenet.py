"""
Feature extraction for Tiny ImageNet using multiple pretrained backbones.
Saves extracted features to disk for efficient experiment iteration.
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
import torchvision.models as models

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.tiny_imagenet import get_tiny_imagenet_loaders


# Backbone configurations: name -> (model_fn, feature_dim)
BACKBONES = {
    'ResNet-18': (lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), 512),
    'ResNet-50': (lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), 2048),
    'MobileNetV3-Small': (lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1), 576),
    'EfficientNet-B0': (lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1), 1280),
}


def create_feature_extractor(model, backbone_name: str):
    """
    Modify model to output features instead of logits.
    """
    if 'ResNet' in backbone_name:
        # Remove the final FC layer
        model.fc = torch.nn.Identity()
    elif 'MobileNet' in backbone_name:
        # Remove classifier
        model.classifier = torch.nn.Identity()
    elif 'EfficientNet' in backbone_name:
        # Remove classifier
        model.classifier = torch.nn.Identity()
    
    return model


def extract_features(model, dataloader, device):
    """Extract features from a dataloader using the given model."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            
            # Handle different output shapes
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)


def get_or_extract_features(backbone_name: str, data_dir: str = './data', 
                            cache_dir: str = './features/tiny_imagenet'):
    """
    Get cached features or extract them if not available.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_dim)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    safe_name = backbone_name.replace(' ', '_').replace('-', '_')
    cache_file = os.path.join(cache_dir, f'{safe_name}_features.npz')
    
    # Check if cached
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        data = np.load(cache_file)
        return (data['X_train'], data['y_train'], 
                data['X_test'], data['y_test'], 
                int(data['dim']))
    
    # Extract features
    print(f"Extracting features using {backbone_name}...")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_fn, feature_dim = BACKBONES[backbone_name]
    model = model_fn()
    model = create_feature_extractor(model, backbone_name)
    model = model.to(device)
    
    # Get data loaders
    train_loader, val_loader = get_tiny_imagenet_loaders(data_dir, batch_size=128)
    
    # Extract features
    start_time = time.time()
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, val_loader, device)
    extract_time = time.time() - start_time
    
    print(f"Feature extraction took {extract_time:.2f}s")
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Cache features
    np.savez(cache_file, 
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test,
             dim=feature_dim)
    print(f"Cached features to {cache_file}")
    
    return X_train, y_train, X_test, y_test, feature_dim


if __name__ == "__main__":
    # Extract features for all backbones
    for backbone_name in BACKBONES.keys():
        print(f"\n{'='*60}")
        print(f"Processing {backbone_name}")
        print('='*60)
        get_or_extract_features(backbone_name)
        print()
