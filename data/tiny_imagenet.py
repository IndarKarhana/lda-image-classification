"""
Tiny ImageNet dataset loader.
200 classes, 100,000 training images (500 per class), 10,000 validation images.
Images are 64x64 RGB.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import urllib.request
import zipfile
import shutil


class TinyImageNet(Dataset):
    """Tiny ImageNet dataset."""
    
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        self.root = root
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(root, 'tiny-imagenet-200')
        
        if download:
            self._download()
        
        self._load_data()
    
    def _download(self):
        """Download and extract Tiny ImageNet if not present."""
        if os.path.exists(self.data_dir):
            return
        
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")
        
        os.makedirs(self.root, exist_ok=True)
        
        if not os.path.exists(zip_path):
            print(f"Downloading Tiny ImageNet from {url}...")
            print("This may take a few minutes (~240 MB)...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete!")
        
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print("Extraction complete!")
        
        # Reorganize validation folder for easier loading
        self._reorganize_val()
        
        # Clean up zip file
        os.remove(zip_path)
    
    def _reorganize_val(self):
        """Reorganize validation set into class folders."""
        val_dir = os.path.join(self.data_dir, 'val')
        val_img_dir = os.path.join(val_dir, 'images')
        val_annotations = os.path.join(val_dir, 'val_annotations.txt')
        
        if not os.path.exists(val_img_dir):
            return  # Already reorganized
        
        # Read annotations
        val_dict = {}
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                val_dict[img_name] = class_id
        
        # Create class folders and move images
        for img_name, class_id in val_dict.items():
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            src = os.path.join(val_img_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # Remove empty images folder
        if os.path.exists(val_img_dir):
            shutil.rmtree(val_img_dir)
    
    def _load_data(self):
        """Load image paths and labels."""
        self.samples = []
        self.class_to_idx = {}
        
        # Load class names (wnids)
        wnids_file = os.path.join(self.data_dir, 'wnids.txt')
        with open(wnids_file, 'r') as f:
            wnids = [line.strip() for line in f]
        
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.classes = wnids
        
        # Load samples
        if self.train:
            train_dir = os.path.join(self.data_dir, 'train')
            for class_id in wnids:
                class_dir = os.path.join(train_dir, class_id, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, self.class_to_idx[class_id]))
        else:
            val_dir = os.path.join(self.data_dir, 'val')
            for class_id in wnids:
                class_dir = os.path.join(val_dir, class_id)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            img_path = os.path.join(class_dir, img_name)
                            self.samples.append((img_path, self.class_to_idx[class_id]))
        
        print(f"Loaded {len(self.samples)} {'training' if self.train else 'validation'} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_tiny_imagenet_loaders(data_dir: str = './data', batch_size: int = 128):
    """
    Get Tiny ImageNet data loaders with appropriate transforms.
    Uses ImageNet normalization since we'll use pretrained ImageNet models.
    """
    # ImageNet normalization (same as pretrained models expect)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms - resize to 224 for pretrained models
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = TinyImageNet(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    val_dataset = TinyImageNet(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for feature extraction
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader = get_tiny_imagenet_loaders('./data')
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels in batch: {len(set(labels.tolist()))}")
